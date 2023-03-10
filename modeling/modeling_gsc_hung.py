from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.utils import make_one_hot
from collections import Counter
from torch_geometric.nn import MessagePassing
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import transformers

###############################################################################
############################### GSC architecture ##############################
###############################################################################
class GSCLayer(MessagePassing):
    def __init__(self, aggr="add"):
        super(GSCLayer, self).__init__(aggr=aggr)

    def forward(self, x, edge_index, edge_embeddings):
        edge_score = self.edge_updater(edge_index=edge_index, x=(x,x), edge_attr=edge_embeddings)
        aggr_out = self.propagate(edge_index, x=(x, x), edge_attr=edge_score) #[N, emb_dim]
        return aggr_out, edge_score
    
    def edge_update(self, x_j, edge_attr):
        edge_score = x_j + edge_attr
        return edge_score 
    
    def message(self, x_j, edge_attr): 
        return edge_attr

class GSC_Message_Passing(nn.Module):
    def __init__(self, k, n_ntype, n_etype, hidden_size, without_regulator, remove_qq, remove_aa, remove_qa, remove_za, remove_zq):
        super().__init__()
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.hidden_size = hidden_size
        self.edge_encoder = nn.Sequential(MLP(n_etype+ n_ntype *2, hidden_size, 1, 1, 0, layer_norm=True), nn.Sigmoid())
        self.k = k
        self.gnn_layers = nn.ModuleList([GSCLayer() for _ in range(k)])
        self.without_regulator = without_regulator
        self.remove_qq = remove_qq
        self.remove_aa = remove_aa
        self.remove_qa = remove_qa
        self.remove_za = remove_za
        self.remove_zq = remove_zq
        if not self.without_regulator:
            self.regulator = MLP(1, hidden_size, 1, 1, 0, layer_norm=True) # can be fold as a * x + b when inference

    def get_graph_edge_embedding(self, edge_index, edge_type, node_type_ids):
        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype) #[E, 39]
        node_type = node_type_ids.view(-1).contiguous() #[`total_n_nodes`, ]
        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        edge_mask = torch.zeros(edge_type.shape[0], dtype=torch.bool, device=edge_type.device)
        if self.remove_za:
            head_mask = head_type == 3
            tail_mask = tail_type == 1
            edge_mask = torch.logical_or(torch.logical_and(head_mask, tail_mask), edge_mask)
        if self.remove_zq:
            head_mask = head_type == 3
            tail_mask = tail_type == 0
            edge_mask = torch.logical_or(torch.logical_and(head_mask, tail_mask), edge_mask)
        edge_emb_copy = edge_embeddings.clone()
        edge_emb_copy[edge_mask] = 0
        return edge_embeddings

    def remove_edge_type(self, edge_index, edge_type, node_type_ids, qq=False, aa=False, qa=False):
        flatten_node_type = node_type_ids.view(-1).contiguous()
        edge_mask = torch.ones(edge_type.shape[0], dtype=torch.bool, device=edge_type.device)
        if qq :
            head_mask = flatten_node_type.index_select(0, edge_index[0]) != 0
            tail_mask = flatten_node_type.index_select(0, edge_index[1]) != 0
            edge_mask = torch.logical_and(torch.logical_or(head_mask, tail_mask), edge_mask)
        if aa :
            head_mask = flatten_node_type.index_select(0, edge_index[0]) != 1
            tail_mask = flatten_node_type.index_select(0, edge_index[1]) != 1
            edge_mask = torch.logical_and(torch.logical_or(head_mask, tail_mask), edge_mask)
        if qa :
            # q -> a
            head_mask = flatten_node_type.index_select(0, edge_index[0]) != 0
            tail_mask = flatten_node_type.index_select(0, edge_index[1]) != 1
            edge_mask = torch.logical_and(torch.logical_or(head_mask, tail_mask), edge_mask)
            # a -> 1
            head_mask = flatten_node_type.index_select(0, edge_index[0]) != 1
            tail_mask = flatten_node_type.index_select(0, edge_index[1]) != 0
            edge_mask = torch.logical_and(torch.logical_or(head_mask, tail_mask), edge_mask)

        edge_index, edge_type = edge_index[:, edge_mask], edge_type[edge_mask]
        return edge_index, edge_type

    def forward(self, adj, node_type_ids):
        _batch_size, _n_nodes = node_type_ids.size()
        n_node_total = _batch_size * _n_nodes
        edge_index, edge_type = adj #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        edge_index, edge_type = self.remove_edge_type(edge_index, edge_type, node_type_ids,
                                                      qq=self.remove_qq, aa=self.remove_aa, qa=self.remove_qa)

        edge_embeddings = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids)
        aggr_out = torch.zeros(n_node_total, 1).to(node_type_ids.device)
        edge_score_list = [edge_embeddings]
        for i in range(self.k):
            # propagate and aggregate between nodes and edges
            aggr_out, edge_score = self.gnn_layers[i](aggr_out, edge_index, edge_embeddings)
            edge_score_list.append(edge_score)
            
        if self.without_regulator:
            aggr_out = aggr_out.view(_batch_size, _n_nodes, -1)
        else:
            aggr_out = self.regulator(aggr_out).view(_batch_size, _n_nodes, -1) # just for normalizing output
        return aggr_out, (edge_index, edge_score_list)


class QAGSC(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim, enc_dim,
                 fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.without_gnn = args.without_gnn
        self.concat_choice = args.concat_choice
        if not self.without_gnn:
            self.gnn = GSC_Message_Passing(k, n_ntype, n_etype, hidden_size=enc_dim, without_regulator=args.without_regulator, remove_qq=args.remove_qq, remove_aa=args.remove_aa, remove_qa=args.remove_qa, remove_za=args.remove_za, remove_zq=args.remove_zq)
        if self.concat_choice:
            self.fc = MLP(sent_dim, fc_dim, 5, n_fc_layer, p_fc, layer_norm=True)
        else :
            self.fc = MLP(sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)


    def forward(self, sent_vecs, concept_ids, node_type_ids, adj, detail=False):
        context_score = self.fc(sent_vecs) # TODO output choice num set to 5 when concat_choice
        if self.concat_choice :
            context_score = context_score.reshape(-1,1)
            sent_vecs = torch.tile(sent_vecs, (1,5)).reshape(-1, sent_vecs.shape[1])
        if not self.without_gnn:
            graph_score, infos = self.gnn(adj, node_type_ids)   #(batch_size, dim_node)
            graph_score = graph_score[:, 0]
            qa_score = context_score + graph_score
        else:
            graph_score = 0
            infos = [None, None, [None, None]]
            qa_score = context_score
        if detail:
            return qa_score, (graph_score, context_score, infos)
        else:
            return qa_score


class LM_QAGSC_HUNG(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, enc_dim,
                 fc_dim, n_fc_layer, p_fc, init_range=0.02, concat_choice=False, encoder_config={}):
        super().__init__()
        self.args = args
        self.init_range = init_range
        self.concat_choice = concat_choice
        decoder_type = QAGSC if 'gsc' in args.counter_type else MRN
        encoder_config.update({"concat_choice":concat_choice, "output_attentions":True})

        with open('data/cpnet/concept_cor.txt', 'r') as f :
            self.ConceptNetID = [x[:-1] for x in f.readlines()]
        self.tokenizer = get_tokenizer("basic_english") 
        self.id_to_token = transformers.RobertaTokenizer.from_pretrained(self.args.encoder).convert_ids_to_tokens
        self.GloVe = GloVe(name='6B', dim=300) #GloVe(name='840B', dim=300)
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = decoder_type(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                            enc_dim, fc_dim, n_fc_layer, p_fc)
        if init_range > 0:
            self.decoder.apply(self._init_weights)

    def match(self, lm_tokens, lm_mask, kg_tokens, kg_types, device):
        bc, d = lm_tokens.shape
        lm_words = [tok[1:] if tok[0]=='Ġ' else tok for tok in self.id_to_token(lm_tokens.flatten())]
        lm_tokens = [self.GloVe.get_vecs_by_tokens(token, lower_case_backup=True) for token in lm_words]
        lm_tokens = torch.stack(lm_tokens).reshape(bc, d, -1).to(device)
        lm_mask = lm_mask.to(device)
        lm_words = np.array(lm_words).reshape(bc, -1)

        bc, d = kg_tokens.shape
        kg_words = [self.ConceptNetID[id] for id in kg_tokens.flatten()]
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
        # kg_tokens = [self.GloVe.get_vecs_by_tokens(token, lower_case_backup=True) for token in kg_words]
        kg_tokens = torch.stack(kg_tokens).reshape(bc, d, -1).to(device)
        kg_mask = torch.logical_or(kg_types == 0, kg_types == 1).to(device)
        kg_words = np.array(kg_words).reshape(bc, -1)

        lines = ['\n']
        for lm_t, lm_w, lm_m, kg_t, kg_w, kg_m in zip(lm_tokens, lm_words, lm_mask, kg_tokens, kg_words, kg_mask) :
            lm_embs = lm_t[lm_m.to(bool)][1:-1]
            kg_embs = kg_t[kg_m]
            lm_word = lm_w[lm_m.to(bool).cpu()][1:-1]
            kg_word = kg_w[kg_m.cpu()]
            scores = torch.matmul(lm_embs, kg_embs.T)
            kg_to_lm = scores.argmax(0)
            for i, word in enumerate(kg_word) :
                lines.append(word + '  :::::  ' + lm_word[kg_to_lm[i]] + '\n')
            with open('match.txt', 'a') as f :
                f.writelines(lines)


    def forward(self, *inputs, detail=False):
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
        sent_vecs, all_hidden_states, (attention_mask, output_mask), attns = self.encoder(*lm_inputs)
        self.match(lm_inputs[0], lm_inputs[1], concept_ids, node_type_ids, device=sent_vecs.device)
        
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]
        if detail:
            logits, infos = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj, detail=detail)
        else:
            logits = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj)
        logits = logits.view(bs, nc)
        if detail:
            return logits, infos
        else:
            return logits
        
    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LM_QAGSC_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
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


