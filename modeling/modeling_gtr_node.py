from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.utils import make_one_hot
from collections import Counter
from torch_geometric.nn import MessagePassing

import copy
import math
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor
import transformers
import wandb
from modeling.multihead_attention import MultiheadAttention

    
class Matcher : # TODO not yet mapped to args
    def __init__(self, encoder, emb_name='840B', emb_dim=300, kg_entity_dir='data/cpnet/concept_cor.txt') :
        from torchtext.data import get_tokenizer
        from torchtext.vocab import GloVe

        with open('data/cpnet/concept_cor.txt', 'r') as f :
            self.KG_entities = [x[:-1] for x in f.readlines()]
        self.LM_tokenizer = transformers.AutoTokenizer.from_pretrained(encoder)
        self.GloVe = GloVe(name=emb_name, dim=emb_dim) # 840B / 6B
        
        self.memory = {}

    def match(self, lm_tokens, lm_mask, kg_tokens, kg_types, qids, device):
        # [lm_inputs[0], lm_inputs[1], concept_ids, node_type_ids]
        # NOTE a bit awkward use of term 'mask' here
        run = False
        if qids is not None:
            for qid in qids :
                if qid not in self.memory :
                    run = True
                    break
        if run :
            bc, d = lm_tokens.shape
            lm_words = [tok[1:] if tok[0]=='Ġ' else tok for tok in \
                        self.LM_tokenizer.convert_ids_to_tokens(lm_tokens.flatten())]
            lm_tokens = [self.GloVe.get_vecs_by_tokens(token, lower_case_backup=True) for token in lm_words]
            lm_tokens = torch.stack(lm_tokens).reshape(bc, d, -1).to(device)
            lm_mask = lm_mask.to(device)
            # lm_words = np.array(lm_words).reshape(bc, -1)

            bc, d = kg_tokens.shape
            kg_words = [self.KG_entities[id] for id in kg_tokens.flatten()]
            kg_tokens = []
            for token in kg_words :
                if token in ['context_node','ab_extra'] :
                    emb = self.GloVe.get_vecs_by_tokens(token, lower_case_backup=True)
                else :
                    emb, count = 0, 0
                    for item in token.split('_') :
                        count += 1
                        emb += self.GloVe.get_vecs_by_tokens(item, lower_case_backup=True)
                    emb /= count
                kg_tokens.append(emb)
            kg_tokens = torch.stack(kg_tokens).reshape(bc, d, -1).to(device)
            # kg_mask = torch.logical_or(kg_types == 0, kg_types == 1).to(device)
            kg_mask = torch.logical_or(kg_types == 0, kg_types != 0).to(device)
            # kg_words = np.array(kg_words).reshape(bc, -1)

            lines = []
            for lm_t, lm_m, kg_t, kg_m in zip(lm_tokens, lm_mask, kg_tokens, kg_mask) :
                lm_embs = lm_t[lm_m.to(bool)][1:-1]
                kg_embs = kg_t[kg_m]
                # lm_word = lm_w[lm_m.to(bool).cpu()][1:-1]
                # kg_word = kg_w[kg_m.cpu()]
                scores = torch.matmul(lm_embs, kg_embs.T)
                kg_to_lm = scores.argmax(0)
                lines.append(kg_to_lm)

            b, c = len(qids), int(bc / len(qids))
            for i, qid in enumerate(qids) :
                self.memory[qid] = lines[i*c : (i+1)*c]
        else :
            lines = []
            if qids is not None:
                for qid in qids :
                    lines = lines + self.memory[qid]
        return lines

###############################################################################
############################### QAGTR architecture ##############################
###############################################################################
class FullTransformer(nn.Module):
    # Adapted from DETR code base : https://github.com/facebookresearch/detr/blob/main/models/transformer.py
    def __init__(self, layer_num, n_ntype, n_etype, d_sentence, d_model, nhead, num_lmtokens, dim_feedforward=2048, 
                 dropout=0.1, scorer_dropout=0.0, 
                 remove_qq=False, remove_aa=False, remove_qa=False, remove_zq=False, remove_za=False, type_embed=True, cls_no_type_embed=False,
                 add_nodetoken=False, add_nodefeat2edgeemb=False, add_nodefeatsim=False, add_textfeat2edgeemb=False,
                 add_textfeatsim=False, add_relativeposemb=False, use_windowattention=False, use_wandb=False, encoder_type='roberta-large', add_qaedge=False, epsilon=1e-8, data=None):
        super().__init__() # TODO layer_num not yet utilized for init
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.remove_qq = remove_qq
        self.remove_aa = remove_aa
        self.remove_qa = remove_qa   
        self.remove_zq = remove_zq
        self.remove_za = remove_za 
        self.use_type_embed = type_embed
        self.cls_no_type_embed = cls_no_type_embed        
        self.add_nodetoken = add_nodetoken
        self.add_nodefeat2edgeemb = add_nodefeat2edgeemb
        self.add_nodefeatsim = add_nodefeatsim
        self.add_textfeat2edgeemb = add_textfeat2edgeemb
        self.add_textfeatsim = add_textfeatsim
        self.add_qaedge = add_qaedge
        self.epsilon = epsilon
        self.use_windowattention = use_windowattention
        self.add_relativeposemb = add_relativeposemb

        # Key Modules
        if type_embed:
            self.type_embed = nn.Embedding(2, d_model)
        self.sent_encoder = nn.Linear(d_sentence, d_model)

        self.layers = nn.ModuleList([])
        for _ in range(layer_num):
            layer = nn.ModuleList([])
            layer.append(nn.LayerNorm(d_model))
            if self.add_relativeposemb:
                layer.append(MultiheadAttention(use_wandb, d_model, nhead, dropout=dropout, batch_first=True, use_relativeposemb=self.add_relativeposemb, num_lmtokens=num_lmtokens))
            elif use_windowattention:
                layer.append(MultiheadAttention(use_wandb, d_model, nhead, dropout=dropout, batch_first=True, use_relativeposemb=self.add_relativeposemb, num_lmtokens=num_lmtokens))
            else:
                layer.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True))
            layer.append(nn.Dropout(dropout))
            layer.append(nn.LayerNorm(d_model))
            layer.append(nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            ))
            layer.append(nn.Dropout(dropout))
            self.layers.append(layer)
        self.graph_scorer = MLP(d_model, d_model, 1, 1, scorer_dropout, layer_norm=True)

        if self.add_nodetoken:
            self.g_type_enc = nn.Embedding(2, d_model)

        self.DIM = 64
        if self.add_relativeposemb or self.add_textfeat2edgeemb or (add_textfeatsim in ["prod","diff"]) or add_textfeatsim in ["prodsum"]:
            self.matcher = Matcher(encoder_type)

        self.activation = F.gelu

    def construct_token(self, node_emb, node_mask, textfeat, qids) :
        """construct token

        Args:
            edge_embeddings (Tensor): edge embeddings (#of edge, C)
            unflattened_edge_type (list): unflattend edge type (20, (edge_type))
            node_emb (Tensor): node embedding (B, N, C)
            node_mask (Tensor): node mask (B, N)

        Returns:
            tokens (Tensor): constucted tokens (B, N, C)
            token_masks (Tensor): constucted tokens (B, N)
            
        """        
        if self.add_relativeposemb:
            matched_ = self.matcher.match(*textfeat, qids=qids, device=node_emb.device)
            matched = (torch.stack(matched_)+1).int().unsqueeze(-1)
        else:
            matched = None
            
        v_num = node_emb.size(1)
        tokens =node_emb
        masks =node_mask
        
        # tokens = torch.cat((tokens, node_emb), 1)
        # masks = torch.cat((masks, node_mask), 1)

        return tokens, masks, v_num, matched

    def get_type_embed(self, bs, lm_tnum, gr_tnum):
        lm_embed, gr_embed = self.type_embed.weight
        return torch.cat([lm_embed.unsqueeze(0).repeat(lm_tnum,1), gr_embed.unsqueeze(0).repeat(gr_tnum,1)], 0).unsqueeze(0).repeat(bs,1,1)

    def forward(self, adj, sent_vecs, node_type_ids, unflattened_edge_type, lm_all_states, lm_mask, textfeat,
                node_emb=None, node_mask=None, qids=None):
        # get edge embeddings                    
        edge_index, edge_type = adj
        
        graph_tokens, graph_mask, v_num, matched_tokens = self.construct_token(node_emb, node_mask, textfeat, qids)
        lm_tokens = self.sent_encoder(lm_all_states)
        lm_mask, lm_output_mask = ~lm_mask[0].to(bool).to(graph_tokens.device), lm_mask[1]
        tgt = torch.cat([lm_tokens, graph_tokens], 1)
        tgt_mask = torch.cat([lm_mask, graph_mask], 1)
        if self.use_type_embed:
            type_embed = self.get_type_embed(tgt.shape[0], lm_tokens.shape[1], graph_tokens.shape[1])

        for layer in self.layers:
            tgt2 = layer[0](tgt)
            if self.use_type_embed:
                qv = tgt2+type_embed
            else:
                qv = tgt2
            if self.add_relativeposemb:
                tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask, matched_index=matched_tokens)
                rpe = layer[1].rel_position_bias
            else:
                if self.use_windowattention:
                    # tgt2, attn = layer[1](qv, qv, value=tgt2, matched_index=None, key_padding_mask=tgt_mask)
                    tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask, matched_index=None)
                    rpe = layer[1].rel_position_bias
                else:
                    tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask)
                    rpe = None
            tgt = tgt + layer[2](tgt2)
            tgt2 = layer[3](tgt)
            tgt2 = layer[4](tgt2)
            tgt = tgt + layer[5](tgt2)
        # score
        graph_score = self.graph_scorer(tgt[:,0,:])

        return graph_score, None, rpe


class QAGTR(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 p_fc, full=False, pretrained_concept_emb=None,
                 freeze_ent_emb=True, p_emb=0.2):
        super().__init__()
        self.args = args
        self.gtr = FullTransformer(layer_num=k,
                                       n_ntype=n_ntype, 
                                       n_etype=n_etype,
                                       d_sentence=sent_dim,
                                       d_model=args.transformer_dim, 
                                       nhead=args.num_heads, 
                                       dim_feedforward=args.transformer_ffn_dim, 
                                       dropout=args.dropouttr,
                                       scorer_dropout=p_fc,
                                       remove_qq=args.remove_qq, 
                                       remove_aa=args.remove_aa, 
                                       remove_qa=args.remove_qa,
                                       add_nodetoken=args.add_nodetoken,
                                       add_nodefeat2edgeemb=args.add_nodefeat2edgeemb,
                                       add_nodefeatsim=args.add_nodefeatsim,
                                       add_textfeat2edgeemb=args.add_textfeat2edgeemb,
                                       add_textfeatsim=args.add_textfeatsim,
                                       add_relativeposemb=args.add_relativeposemb,
                                       use_windowattention=args.use_windowattention,
                                       encoder_type=args.encoder,
                                       remove_za=args.remove_za, 
                                       remove_zq=args.remove_zq,
                                       type_embed=not args.without_type_embed,
                                       cls_no_type_embed=args.cls_without_type_embed,
                                       num_lmtokens=args.max_seq_len,
                                       use_wandb=args.use_wandb,
                                       data=args.dataset)


    def forward(self, sent_vecs, concept_ids, node_type_ids, adj, adj_lengths, unflattened_edge_type, lm_all_states, 
                lm_mask, textfeat, node_emb=None, node_mask=None, qids=None, detail=False, emb_data=None, context_mask=None):
        # node_emb, node_mask = None, None
        # vanilla Full-Transformer does not work without these dummy variables
            # node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)
        
        if self.args.detach_lm :
            lm_all_states = lm_all_states.detach()

        graph_score, context_score = 0,0 
        qa_score, infos, rpe = self.gtr(adj, sent_vecs, node_type_ids, unflattened_edge_type, lm_all_states, 
                                    lm_mask, textfeat, node_emb, node_mask, qids=qids)

        if detail:
            return qa_score, (graph_score, context_score, infos), rpe
        else:
            return qa_score, rpe


class LM_QAGTR(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, 
                 fc_dim, n_fc_layer, p_fc,
                 n_concept=None, concept_dim=None, concept_in_dim=None,
                 init_range=0.02, encoder_config={},
                 pretrained_concept_emb=None, freeze_ent_emb=True, p_emb=0.2):
        super().__init__()
        self.args = args
        self.init_range = init_range
        self.full_decoder = args.decoder_type=='transformer_concat'
        self.add_qaedge = args.add_qaedge
        self.use_nodeemb = args.add_qaedge or args.add_nodetoken or args.add_nodefeat2edgeemb or (args.add_nodefeatsim in ["diff", "prod"])

        self.encoder = TextEncoder(model_name, **encoder_config)
        if args.add_qaedge:
            n_etype = n_etype+2
        self.decoder = QAGTR(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                            p_fc, full=self.full_decoder)

        if init_range > 0:
            self.decoder.apply(self._init_weights)

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                            use_contextualized=False, concept_in_dim=concept_in_dim,
                                            pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.dropout_e = nn.Dropout(p_emb)
        self.concept_dim = concept_dim

    def forward(self, *inputs, qids=None, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]
        
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        sent_vecs, lm_all_states, lm_mask = self.encoder(*lm_inputs)
        unflattened_edge_type = edge_type
        
        gnn_input0 = torch.zeros((concept_ids.size(0), 1, self.concept_dim), device=node_type_ids.device) #Context Node
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, None)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        node_emb = torch.cat((gnn_input0, gnn_input1), dim=1) #(B, N, D)
        node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1), node_type_ids, node_emb, node_mask)
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        textfeat = None
        if self.args.add_textfeat2edgeemb or self.args.add_textfeatsim != 'none' or self.args.add_relativeposemb:
            textfeat = [lm_inputs[0], lm_inputs[1], concept_ids, node_type_ids] # [20,88], [20,88], [20,32], [20,32]

        if detail:
            logits, infos, rpe = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj, 
                                         adj_lengths, unflattened_edge_type, lm_all_states.to(node_type_ids.device), 
                                         lm_mask, textfeat, node_emb=node_emb, node_mask=node_mask, qids=qids, detail=detail)
        else:
            logits, rpe = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj,
                                  adj_lengths, unflattened_edge_type, lm_all_states.to(node_type_ids.device), 
                                  lm_mask, textfeat, node_emb=node_emb, node_mask=node_mask, qids=qids)
        logits = logits.view(bs, nc)
        if detail:
            return logits, infos, rpe
        else:
            return logits
        
    def batch_graph(self, edge_index_init, edge_type_init, n_nodes, node_type, node_emb, node_mask):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index, edge_type = list(), list()

        if node_emb is not None:
            node_emb_withmask = node_emb * (~node_mask).unsqueeze(-1)
            node_sim = node_emb_withmask@node_emb_withmask.transpose(-1, -2)

        for _i_ in range(n_examples):
            if node_emb is not None:
                new_edge_index, new_edge_type = self.transform_graph(edge_index_init[_i_], edge_type_init[_i_], n_nodes, node_type[_i_], node_sim[_i_], node_mask[_i_]) 
            else:
                new_edge_index, new_edge_type = self.transform_graph(edge_index_init[_i_], edge_type_init[_i_], n_nodes, node_type[_i_]) 
                
            edge_index.append(new_edge_index + _i_ * n_nodes)
            edge_type.append(new_edge_type)

        # edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]

        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type, dim=0) #[total_E, ]
        return edge_index, edge_type
    
    def transform_graph(self, edge_index, edge_type, n_nodes, node_type, node_sim=None, node_mask=None):
        # add qa edge (if not connected) -> to (q-a) fully connected
        if self.add_qaedge:
            sim_sparse = node_sim.to_sparse()
            sim_indices = sim_sparse.indices()
            # topk_indices = sim_sparse.values().topk(math.floor((~node_mask).sum().item()*0.2)).indices
            # bottomk_indices = sim_sparse.values().topk(math.floor((~node_mask).sum().item()*0.2), largest=False).indices
            topk_indices = sim_sparse.values().topk(1).indices
            bottomk_indices = sim_sparse.values().topk(1, largest=False).indices

            if topk_indices.numel() != 0:
                topk_edge_index = sim_indices.index_select(1, topk_indices)
                topk_edge_type = torch.ones(topk_edge_index.size(1), dtype=torch.int64, device=edge_type.device)*38
                new_edge_index = torch.cat((edge_index, topk_edge_index), 1)
                new_edge_type = torch.cat((edge_type, topk_edge_type))
            else:
                new_edge_index = edge_index
                new_edge_type = edge_type
            
            if bottomk_indices.numel() !=0:
                bottomk_edge_index = sim_indices.index_select(1, bottomk_indices)
                bottomk_edge_type = torch.ones(bottomk_edge_index.size(1), dtype=torch.int64, device=edge_type.device)*39
                new_edge_index = torch.cat((new_edge_index, bottomk_edge_index), 1)
                new_edge_type = torch.cat((new_edge_type, bottomk_edge_type))
            else:
                new_edge_index = new_edge_index
                new_edge_type = new_edge_type
        else:
            new_edge_index = edge_index
            new_edge_type = edge_type

        return new_edge_index, new_edge_type

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LM_QAGSC_DataLoader(object):

    def __init__(self, args,  train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        if 'aristo-roberta' in model_name:
            model_type = 'aristo-roberta'
        else:
            model_type = MODEL_NAME_TO_CLASS[model_name]
        print ('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, args.load_sentvecs_model_path)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, args.load_sentvecs_model_path)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, args.load_sentvecs_model_path)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        print('max train seq length: ', self.train_encoder_data[1].sum(dim=2).max().item())
        print('max dev seq length: ', self.dev_encoder_data[1].sum(dim=2).max().item())
        if test_statement_path is not None:
            print('max test seq length: ', self.test_encoder_data[1].sum(dim=2).max().item())

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)


    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)

