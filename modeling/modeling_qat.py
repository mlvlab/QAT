from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils_path import *
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
        # NOTE a bit awkward use of term 'mask' here
        run = False
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
            for qid in qids :
                lines = lines + self.memory[qid]
        return lines

###############################################################################
############################### QAGTR architecture ##############################
###############################################################################



class FullTransformer(nn.Module):
    # Adapted from DETR code base : https://github.com/facebookresearch/detr/blob/main/models/transformer.py
    def __init__(self, layer_num, n_ntype, n_etype, d_sentence, d_model, nhead, num_lmtokens, dim_feedforward=2048, 
                 dropout=0.1, scorer_dropout=0.0, add_nodefeatsim=False,
                 cls_no_type_embed=False, encoder_type='roberta-large', epsilon=1e-8, data=None, rpe_2=False, drop_ratio=1.0):
        super().__init__() # TODO layer_num not yet utilized for init
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.cls_no_type_embed = cls_no_type_embed        
        self.epsilon = epsilon
        self.drop_ratio = drop_ratio
        self.add_nodefeatsim=add_nodefeatsim

        # Key Modules
        self.type_embed = nn.Embedding(2, d_model)

        self.sent_encoder = nn.Linear(d_sentence, d_model)
        print(encoder_type)


        self.layers = nn.ModuleList([])
        for _ in range(layer_num):
            layer = nn.ModuleList([])
            layer.append(nn.LayerNorm(d_model))
            # if self.add_relativeposemb:
            layer.append(MultiheadAttention(False, d_model, nhead, dropout=dropout, batch_first=True, use_relativeposemb=True, num_lmtokens=num_lmtokens, rpe_2=rpe_2))
            # elif self.use_windowattention:
                # layer.append(MultiheadAttention(False, d_model, nhead, dropout=dropout, batch_first=True, use_relativeposemb=self.add_relativeposemb, num_lmtokens=num_lmtokens))
            # else:
                # layer.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True))
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
            
        self.qa_scorer = MLP(d_model, d_model, 1, 1, scorer_dropout, layer_norm=True)

        # if self.add_nodetoken:
            # self.g_type_enc = nn.Embedding(2, d_model)

        if self.add_nodefeatsim:
            self.DIM = 4
            self.node_enc = nn.Linear(d_model, self.DIM)
        else:
            self.DIM = 0
            
        self.matcher = Matcher(encoder_type)
        self.edge_encoder = nn.Sequential(MLP(n_etype+ n_ntype *2 + self.DIM, d_model, d_model, 1, scorer_dropout, layer_norm=True))
        
        if data in ["obqa", "csqa"]:
            self.path_encoder = MLP(45+self.DIM, d_model, d_model, 1, 0, layer_norm=True)
        else:
            self.path_encoder = MLP(41+self.DIM, d_model, d_model, 1, 0, layer_norm=True)

        self.activation = F.gelu

    def get_graph_edge_embedding(self, edge_index, edge_type, node_type_ids, node_emb, textfeat, qids):
        """construct edge embedidngs

        Args:
            edge_index (Tensor): edge_index (2, # of edges)
            edge_type (Tensor): edge_type (# of edges,)
            node_type_ids (_type_): _description_
            node_emb (Tensor) : node embedding (b, n, c)
        Returns:
            edge_embeddings (Tensor): (# of edges, c)
        """        
        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype) #[E, 39]
        node_type = node_type_ids.view(-1).contiguous() #[`total_n_nodes`, ]
        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        
        if self.add_nodefeatsim:
            # add node feature similarity to edge embeddings
            flattened_node_emb = node_emb.view(-1, node_emb.size(-1)).contiguous()
            head_feat = flattened_node_emb[edge_index[0]]
            tail_feat = flattened_node_emb[edge_index[1]]
            # elif self.add_nodefeatsim == 'diff':
            sim = self.node_enc(tail_feat - head_feat)
            edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec, sim], dim=1)) #[E+N, emb_dim]
        else:
            edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        # if self.add_relativeposemb:
        matched_ = self.matcher.match(*textfeat, qids=qids, device=edge_vec.device)
        matched = (torch.stack(matched_) + 1).int()
        flattened_matched = matched.view(-1)
        head_matched = flattened_matched[edge_index[0]]
        tail_matched = flattened_matched[edge_index[1]]
        matched = torch.stack([head_matched, tail_matched], dim=-1)
        
        return edge_embeddings, matched

    def get_graph_path_embedding(self, metapath_feature, metapath_feature_count, node_emb, textfeat, qids):
        """construct edge embedidngs

        Args:
            metapath_feature (list): list of metapath_feature (20, # of paths)
            node_emb (Tensor) : node embedding (b, n, c)
        Returns:
            path_embeddings (Tensor): (# of edges, c)
        """        
        #Prepare path feature
        flattened_metapath_feature = torch.cat(metapath_feature, 0)
        path_idx = torch.cat(metapath_feature_count, 1).long()
        if self.add_nodefeatsim:
            flattened_node_emb = node_emb.view(-1, node_emb.size(-1)).contiguous()
            head_feat = flattened_node_emb[path_idx[0]]
            tail_feat = flattened_node_emb[path_idx[1]]
            sim = self.node_enc(tail_feat - head_feat)
            path_embeddings = self.path_encoder(torch.cat([flattened_metapath_feature, sim], dim=1)) #[E+N, emb_dim]
        else:
            path_embeddings = self.path_encoder(flattened_metapath_feature) #[E+N, emb_dim]        

        matched_ = self.matcher.match(*textfeat, qids=qids, device=flattened_metapath_feature.device)
        matched = (torch.stack(matched_) + 1).int()
        flattened_matched = matched.view(-1)
        head_matched = flattened_matched[path_idx[0]]
        tail_matched = flattened_matched[path_idx[1]]
        matched_path = torch.stack([head_matched, tail_matched], dim=-1)

        return path_embeddings, matched_path
        # else:
            # return path_embeddings


    def construct_token(self, edge_embeddings, path_embeddings, unflattened_edge_type, node_emb, node_mask, metapath_feature, metapath_feature_count, matched=None, matched_path=None) :
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
        token_lens = [len(graph) for graph in unflattened_edge_type]
        path_lens = [len(mp) for mp in metapath_feature]
        max_len = max(token_lens) 
        path_max_len = max(path_lens)
        total_len = max_len + path_max_len
        tokens, masks, matched_tokens, matched_path_tokens = [], [], [], []
        path_tokens, path_masks = [], []
        start, path_start = 0, 0
        idx = 0
        
        for length, path_len in zip(token_lens, path_lens) :
            emb = edge_embeddings[start:start+length]
            emb = F.pad(emb, (0,0,0,max_len-length))
            path_emb = path_embeddings[path_start:path_start+path_len]
            path_emb = F.pad(path_emb, (0,0,0, path_max_len-path_len))
            tokens.append(emb)
            path_tokens.append(path_emb)
            # if matched is not None:
            matched_ = matched[start:start+length]
            matched_ = F.pad(matched_, (0,0,0,max_len-length))
            matched_tokens.append(matched_)          
            # if matched_path is not None:
            matched_path_ = matched_path[path_start:path_start+path_len]    
            matched_path_ = F.pad(matched_path_, (0,0,0, path_max_len-path_len))
            matched_path_tokens.append(matched_path_)

            mask = torch.zeros(emb.shape[0], dtype=bool, device=emb.device)
            path_mask = torch.zeros(path_emb.shape[0], dtype=bool, device=path_emb.device)
            # Drop MP
            if self.training and self.drop_ratio != 1:            
                if metapath_feature_count[idx][0,0] == 0 and metapath_feature_count[idx][1,0] == 0:
                    path_len = 0
                    
                rand_idx = torch.randperm(length, device=emb.device)
                rand_idx = rand_idx[:math.floor((length)*self.drop_ratio)]
                mask[rand_idx] = True
                
                if path_len > 0:
                    rand_idx_path = torch.randperm(path_len, device=emb.device)
                    rand_idx_path = rand_idx_path[:math.floor((path_len)*self.drop_ratio)]
                    if math.floor((path_len)*self.drop_ratio) > 0:   
                        path_mask[rand_idx_path] = True
                         
            mask[length:] = True 
            path_mask[path_len:] = True
            
            masks.append(mask)
            path_masks.append(path_mask)
            start += length
            path_start += path_len
            idx += 1
        tokens = torch.stack(tokens)
        masks = torch.stack(masks)
        path_tokens = torch.stack(path_tokens)
        path_masks =torch.stack(path_masks)

        # if matched is not None:
        matched_tokens = torch.stack(matched_tokens)        
        # if matched_path is not None:
        matched_paths = torch.stack(matched_path_tokens)  
        matched_tokens = torch.concat((matched_tokens, matched_paths), 1)      

        e_num = tokens.size(1)
        p_num = path_tokens.size(1)
        
        if node_emb is not None:
            v_num = node_emb.size(1)
        else:
            v_num = 0

        tokens = torch.cat((tokens, path_tokens), 1)
        masks = torch.cat((masks, path_masks), 1)

        # if self.add_relativeposemb:
        return tokens, masks, e_num, v_num, p_num, matched_tokens
        # else:
            # return tokens, masks, e_num, v_num, p_num

    def get_type_embed(self, bs, lm_tnum, gr_tnum):
        lm_embed, gr_embed = self.type_embed.weight
        return torch.cat([lm_embed.unsqueeze(0).repeat(lm_tnum,1), gr_embed.unsqueeze(0).repeat(gr_tnum,1)], 0).unsqueeze(0).repeat(bs,1,1)

    def get_graph_type(self, bs, e_num, v_num, lm_num):
        v_embed, e_embed = self.g_type_enc.weight
        return torch.cat([torch.zeros(lm_num, e_embed.size(0), device=e_embed.device),e_embed.unsqueeze(0).repeat(e_num,1), v_embed.unsqueeze(0).repeat(v_num,1)], 0).unsqueeze(0).repeat(bs,1,1)

    def forward(self, adj, sent_vecs, node_type_ids, unflattened_edge_type, lm_all_states, lm_mask, textfeat,
                metapath_feature, metapath_feature_count, node_emb=None, node_mask=None, qids=None):
        # get edge embeddings                    
        edge_index, edge_type = adj
        
        # if self.add_relativeposemb:
        edge_embeddings, matched = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids, node_emb, textfeat, qids)
        path_embeddings, matched_path = self.get_graph_path_embedding(metapath_feature, metapath_feature_count, node_emb, textfeat, qids)
        graph_tokens, graph_mask, e_num, v_num, p_num, matched_tokens = self.construct_token(edge_embeddings, path_embeddings, unflattened_edge_type, node_emb, node_mask, metapath_feature, metapath_feature_count, matched, matched_path)
        # else:
            # edge_embeddings = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids, node_emb, lm_all_states, textfeat, qids)
            # path_embeddings = self.get_graph_path_embedding(metapath_feature, metapath_feature_count, textfeat, qids)
            # graph_tokens, graph_mask, e_num, v_num, p_num = self.construct_token(edge_embeddings, path_embeddings, unflattened_edge_type, node_emb, node_mask, metapath_feature, metapath_feature_count, matched=None)

        lm_tokens = self.sent_encoder(lm_all_states)
        lm_mask, lm_output_mask = ~lm_mask[0].to(bool).to(graph_tokens.device), lm_mask[1]
        
        tgt = torch.cat([lm_tokens, graph_tokens], 1)
        tgt_mask = torch.cat([lm_mask, graph_mask], 1)

        type_embed = self.get_type_embed(tgt.shape[0], lm_tokens.shape[1], graph_tokens.shape[1])
        # if self.add_nodetoken:
            # g_type_embed = self.get_graph_type(tgt.shape[0], e_num, v_num, lm_tokens.size(1))

        for layer in self.layers:
            tgt2 = layer[0](tgt)
            qv = tgt2+type_embed
            # if self.add_relativeposemb:
            tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask, matched_index=matched_tokens, rpe_2=True)
            rpe = layer[1].rel_position_bias
            # else:
                # if self.use_windowattention:
                    # tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask, matched_index=None)
                    # rpe = layer[1].rel_position_bias
                # else:
                    # tgt2, attn = layer[1](qv, qv, value=tgt2, key_padding_mask=tgt_mask)
                    # rpe = None
            tgt = tgt + layer[2](tgt2)
            tgt2 = layer[3](tgt)
            tgt2 = layer[4](tgt2)
            tgt = tgt + layer[5](tgt2)
        # score
        graph_score = self.qa_scorer(tgt[:,0,:])

        return graph_score, rpe


class QAT(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 p_fc, pretrained_concept_emb=None,
                 freeze_ent_emb=True, p_emb=0.2):
        super().__init__()
        self.args = args
        self.qat = FullTransformer(layer_num=k,
                                       n_ntype=n_ntype, 
                                       n_etype=n_etype,
                                       d_sentence=sent_dim,
                                       d_model=args.transformer_dim, 
                                       nhead=args.num_heads, 
                                       num_lmtokens=args.max_seq_len,
                                       dim_feedforward=args.transformer_ffn_dim, 
                                       dropout=args.dropouttr,
                                       scorer_dropout=p_fc,
                                       add_nodefeatsim=True if self.args.dataset == 'medqa_usmle' else False,
                                       encoder_type=args.encoder,
                                       cls_no_type_embed=args.cls_without_type_embed,
                                       data=args.dataset,
                                       rpe_2=args.rpe_2,
                                       drop_ratio=args.drop_ratio)


    def forward(self, sent_vecs, concept_ids, node_type_ids, adj, metapath_feature, metapath_feature_count, adj_lengths, unflattened_edge_type, lm_all_states,
                lm_mask, textfeat, node_emb=None, node_mask=None, qids=None):
        # node_emb, node_mask=None, None        
        if self.args.detach_lm :
            lm_all_states = lm_all_states.detach()

        qa_score, rpe = self.qat(adj, sent_vecs, node_type_ids, unflattened_edge_type, lm_all_states, 
                                   lm_mask, textfeat, metapath_feature, metapath_feature_count, node_emb, node_mask, qids=qids)

        return qa_score, rpe


class LM_QAT(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, 
                 fc_dim, n_fc_layer, p_fc,
                 n_concept=None, concept_dim=None, concept_in_dim=None,
                 init_range=0.02, encoder_config={},
                 pretrained_concept_emb=None, freeze_ent_emb=True, p_emb=0.2):
        super().__init__()
        self.args = args
        self.init_range = init_range

        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAT(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                            p_fc)

        if init_range > 0:
            self.decoder.apply(self._init_weights)

        if self.args.dataset == 'medqa_usmle':
            self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                                use_contextualized=False, concept_in_dim=concept_in_dim,
                                                pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
            self.dropout_e = nn.Dropout(p_emb)
            self.concept_dim = concept_dim

    def forward(self, *inputs, qids=None):
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
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-8]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-8:-4]] + [sum(x,[]) for x in inputs[-4:]]
        
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count, edge_index, edge_type = _inputs
        sent_vecs, lm_all_states, lm_mask = self.encoder(*lm_inputs)
        unflattened_edge_type = edge_type
        
        if self.args.dataset == 'medqa_usmle':
            gnn_input0 = torch.zeros((concept_ids.size(0), 1, self.concept_dim), device=node_type_ids.device) #Context Node
            gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, None)
            gnn_input1 = gnn_input1.to(node_type_ids.device)
            node_emb = torch.cat((gnn_input0, gnn_input1), dim=1) #(B, N, D)
            node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)
        else:
            node_emb, node_mask = None, None
        
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1), node_type_ids, node_emb, node_mask)
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        # if self.args.add_relativeposemb:
        textfeat = [lm_inputs[0], lm_inputs[1], concept_ids, node_type_ids]

        logits, rpe = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj,
                                metapath_feature, metapath_feature_count,
                                adj_lengths, unflattened_edge_type, lm_all_states.to(node_type_ids.device),
                                lm_mask, textfeat, node_emb=node_emb, node_mask=node_mask, qids=qids)
        logits = logits.view(bs, nc)

        return logits, rpe
        
    def batch_graph(self, edge_index_init, edge_type_init, n_nodes, node_type, node_emb, node_mask):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index, edge_type = list(), list()

        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_type = [edge_type_init[_i_] for _i_ in range(n_examples)]

        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type, dim=0) #[total_E, ]
        return edge_index, edge_type
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LM_QAT_DataLoader(object):

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
        *self.train_decoder_data, self.train_metapath, self.train_adj_data, self.metapath_fea_size = load_sparse_adj_data_and_metapathonehot_with_contextnode_changed(train_adj_path, max_node_num, num_choice, args)

        print(len(self.train_metapath))
        *self.dev_decoder_data, self.dev_metapath, self.dev_adj_data, self.metapath_fea_size = load_sparse_adj_data_and_metapathonehot_with_contextnode_changed(dev_adj_path, max_node_num, num_choice, args)

        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, args.load_sentvecs_model_path)
            *self.test_decoder_data, self.test_metapath, self.test_adj_data, self.metapath_fea_size = load_sparse_adj_data_and_metapathonehot_with_contextnode_changed(test_adj_path, max_node_num, num_choice, args)

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
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data, metapath_data=self.train_metapath)


    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data, metapath_data=self.dev_metapath)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data, metapath_data=self.train_metapath)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data, metapath_data=self.test_metapath)

