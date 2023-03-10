from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.utils import make_one_hot
from collections import Counter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


###############################################################################
############################### GSC architecture ##############################
###############################################################################
class GSCLayer(MessagePassing):
    def __init__(self, aggr="mean", edge_enc_gnn=False, input_dim=0, hidden_dim=0):
        super(GSCLayer, self).__init__(aggr=aggr)
        self.edge_enc_gnn = edge_enc_gnn
        if edge_enc_gnn:
            self.edge_encoder = nn.Sequential(MLP(input_dim, hidden_dim, 1, 1, 0, layer_norm=True))
            
    def forward(self, x, edge_index, edge_embeddings):
        if self.edge_enc_gnn:
            edge_embeddings = self.edge_encoder(edge_embeddings)
        aggr_out = self.propagate(edge_index, x=x, edge_weight=edge_embeddings) #[N, emb_dim]
        return aggr_out, edge_embeddings
    
    def message(self, x_j, edge_weight): 
        return x_j*edge_weight.view(-1,1)

class GSC_Message_Passing(nn.Module):
    def __init__(self, k, n_ntype, n_etype, hidden_size, without_regulator, remove_qq, remove_aa, remove_qa, remove_za, remove_zq, edge_enc_gnn):
        super().__init__()
        self.n_ntype = n_ntype
        self.n_etype = n_etype+1
        self.hidden_size = hidden_size
        self.edge_enc_gnn = edge_enc_gnn
        if not edge_enc_gnn:
            self.edge_encoder = nn.Sequential(MLP(self.n_etype+ n_ntype *2, hidden_size, 1, 1, 0, layer_norm=True))
        self.k = k
        self.gnn_layers = nn.ModuleList([GSCLayer(edge_enc_gnn=edge_enc_gnn, input_dim=self.n_etype+n_ntype*2, hidden_dim=hidden_size) for _ in range(k)])
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
        # edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]
        edge_embeddings = torch.cat([edge_vec, headtail_vec], dim=1)

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
        # Generate self-loops 
        edge_index = add_self_loops(edge_index)[0]
        selfloop_type = torch.ones(edge_index[0].unique().numel(), dtype=edge_type.dtype, device=edge_index.device)*(self.n_etype-1)
        edge_type = torch.cat((edge_type, selfloop_type))

        edge_embeddings = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids)
        if not self.edge_enc_gnn:
            edge_embeddings = self.edge_encoder(edge_embeddings)
        aggr_out = torch.ones(n_node_total, 1).to(node_type_ids.device)
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
        if not self.without_gnn:
            self.gnn = GSC_Message_Passing(k, n_ntype, n_etype, hidden_size=enc_dim, without_regulator=args.without_regulator, remove_qq=args.remove_qq, remove_aa=args.remove_aa, remove_qa=args.remove_qa, remove_za=args.remove_za, remove_zq=args.remove_zq, edge_enc_gnn=args.edge_enc_gnn)
        self.fc = MLP(sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)


    def forward(self, sent_vecs, concept_ids, node_type_ids, adj, detail=False):
        context_score = self.fc(sent_vecs)
        if not self.without_gnn:
            graph_score, infos = self.gnn(adj, node_type_ids)   #(batch_size, dim_node)
            answer_mask = (node_type_ids==1).long().unsqueeze(-1)
            graph_score = (graph_score * answer_mask).sum(1)
            # graph_score = graph_score[:, 0]
            qa_score = context_score + graph_score
        else:
            graph_score = 0
            infos = [None, None, [None, None]]
            qa_score = context_score
        if detail:
            return qa_score, (graph_score, context_score, infos)
        else:
            return qa_score


class LM_QAGSC(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, enc_dim,
                 fc_dim, n_fc_layer, p_fc, init_range=0.02, encoder_config={}):
        super().__init__()
        self.args = args
        self.init_range = init_range
        decoder_type = QAGSC if 'gsc' in args.counter_type else MRN
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = decoder_type(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                            enc_dim, fc_dim, n_fc_layer, p_fc)
        if init_range > 0:
            self.decoder.apply(self._init_weights)

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
        sent_vecs, _ = self.encoder(*lm_inputs)
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



###############################################################################
############################ Hard counter with MLP ############################
###############################################################################

class MRN(nn.Module): 
    def __init__(self, args, k, n_ntype, n_etype, sent_dim, enc_dim, fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.fc = MLP(sent_dim, fc_dim, 1, 0, p_fc, layer_norm=True)
        if args.counter_type == '1hop':
            self.mlp = MLP(n_etype * n_ntype ** 2, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
        elif args.counter_type == '2hop':
            self.mlp = MLP(n_etype ** 2 * n_ntype ** 3 + n_etype * n_ntype ** 2, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
        else:
            raise NotImplementedError

    def forward(self, sent_vecs, edge_counts, node_type_ids, adj):
        # we cache the multihop relation counting results of the graph for efficiency
        # you may see the cache process in line 208-253 of utils/data_utils.py 
        return self.mlp(edge_counts.to(node_type_ids.device)) + self.fc(sent_vecs)
