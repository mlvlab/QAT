import random

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling.modeling_qat import *
from modeling.medqa_dataset import MedQA_DataLoader
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
import torch.nn.functional as F
import numpy as np

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data, qids=qids)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)

    return n_correct / n_samples


def main():
    #TODO: Check useless parser
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qat/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--load_sentvecs_model_path', default=None)
    parser.add_argument('--without_amp', dest='without_amp', action='store_true', help='disable mixed precision training')

    parser.add_argument('--edge_emb_size', type=int, default=1)
    parser.add_argument('--detach_lm', action='store_true')
    parser.add_argument('--epsilon', type=float, default=1e-8)

    parser.add_argument('--lambda_rpe', type=float, default=1.)
    parser.add_argument('--inverse_relation', action='store_true')
    parser.add_argument('--rpe_2', action='store_true')
    parser.add_argument('-k', type=int, default=2)

    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    
    parser.add_argument('--train_path', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_path', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_path', default=f'data/{args.dataset}/graph/test.graph.adj.pk')

    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('--fc_dim', default=512, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')

    # * Transformer Decoder
    parser.add_argument('--transformer_dim', type=int, default=1024, help='transformer decoder model dim')
    parser.add_argument('--transformer_ffn_dim', type=int, default=1024, help='transformer decoder ffn dim')
    parser.add_argument('--num_heads', type=int, default=8, help='transformer decoder head num')
    parser.add_argument('--without_type_embed', action='store_true')
    parser.add_argument('--cls_without_type_embed', action='store_true')

    parser.add_argument('--max_node_num', default=32, type=int) 
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
 

    # regularization
    parser.add_argument('--dropoutf', type=float, default=0.0, help='dropout for fully-connected layers')
    parser.add_argument('--dropouttr', type=float, default=0.1, help='dropout for transformer decoder')
    parser.add_argument('--drop_ratio', type=float, default=1.0)

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=2, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    # Added for Medical QA
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_acc,test_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    if args.dataset == 'medqa_usmle':
        cp_emb = [np.load(path) for path in args.ent_emb_paths]
        cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        print('| num_concepts: {} |'.format(concept_num))
    else:
        concept_num, concept_dim, cp_emb = 0, 0, None
        
    
    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("Use two devices")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
        
    if args.dataset == 'medqa_usmle' :
        dataset = MedQA_DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=(device0, device1),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, use_cache=args.use_cache)
    else :
        dataset = LM_QAT_DataLoader(args, args.train_statements, args.train_adj,
                                                args.dev_statements, args.dev_adj,
                                                args.test_statements, args.test_adj,
                                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                                device=(device0, device1),
                                                model_name=args.encoder,
                                                max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                                subsample=args.subsample, use_cache=args.use_cache)
    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    model = LM_QAT(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, 
            fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num, p_fc=args.dropoutf, init_range=args.init_range, pretrained_concept_emb=cp_emb,
            n_concept=concept_num, concept_dim=args.transformer_dim, concept_in_dim=concept_dim)

    model.encoder.to(device0)
    model.decoder.to(device1)
    if (args.add_nodefeatsim in ["diff", "prod"]):
        model.concept_emb.to(device1)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise NotImplementedError

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    # Creates once at the beginning of training
    if not args.without_amp:
        scaler = torch.cuda.amp.GradScaler()
    freeze_net(model.encoder)
    for epoch_id in range(args.n_epochs):
        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder)
            
        model.train()
        for qids, labels, *input_data in dataset.train():
            optimizer.zero_grad()
            bs = labels.size(0)
            if not args.without_amp:
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        logits, rpe = model(*[x[a:b] for x in input_data], qids=qids[a:b])
                        loss = loss_func(logits, labels[a:b])
                        loss -= rpe.tanh().mean() * args.lambda_rpe
                        loss = loss * (b - a) / bs
                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    scaler.scale(loss).backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                scheduler.step()
            else:
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits, rpe = model(*[x[a:b] for x in input_data], qids=qids[a:b])
                    loss = loss_func(logits, labels[a:b])
                    loss -= rpe.tanh().mean() * args.lambda_rpe
                    loss = loss * (b - a) / bs
                    loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # scheduler.step() # change the order due to the warning
                optimizer.step()
                scheduler.step()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                total_loss = 0
                start_time = time.time()
            global_step += 1

        model.eval()
        dev_acc = evaluate_accuracy(dataset.dev(), model)
        if not args.save_model:
            test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
        else:
            eval_set = dataset.test()
            total_acc = []
            count = 0
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(eval_set):
                        count += 1
                        logits, _ = model(*input_data, qids=qids)
                        predictions = logits.argmax(1) #[bsize, ]
                        preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                        for i, (qid, label, pred, _preds_ranked) in enumerate(zip(qids, labels, predictions, preds_ranked)):
                            acc = int(pred.item()==label.item())
                            print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            f_preds.flush()
                            total_acc.append(acc)
            test_acc = float(sum(total_acc))/len(total_acc)

        print('-' * 71)
        print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
        print('-' * 71)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id
            print('| epoch {:3} | step {:5} | best_dev_acc {:7.4f} | final_test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, final_test_acc))
            if args.save_model:
                torch.save([model, args], model_path)
                with open(model_path +".log.txt", 'w') as f:
                    for p in model.named_parameters():
                        print (p, file=f)
                print(f'model saved to {model_path}')
        
        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    model_state, old_args = torch.load(model_path, map_location=torch.device('cpu'))

    model = LM_QAT(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation,
            fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num, p_fc=old_args.dropoutf, init_range=old_args.init_range,
            pretrained_concept_emb=cp_emb, concept_in_dim=concept_dim, concept_dim=old_args.transformer_dim, n_concept=concept_num)

    model.load_state_dict(model_state.state_dict(), strict=False)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    print ('inhouse?', args.inhouse)
    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    if args.dataset == 'medqa_usmle' :
        dataset = MedQA_DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=(device0, device1),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, use_cache=args.use_cache)
    else :
        dataset = LM_QAT_DataLoader(args, args.train_statements, args.train_adj,
                                                args.dev_statements, args.dev_adj,
                                                args.test_statements, args.test_adj,
                                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                                device=(device0, device1),
                                                model_name=args.encoder,
                                                max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                                subsample=args.subsample, use_cache=args.use_cache)    
    dev_acc = evaluate_accuracy(dataset.dev(), model) if args.test_statements else 0.0
    print('-' * 71)
    print('dev_acc {:7.4f}'.format(dev_acc))
    test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
    print('test_acc {:7.4f}'.format(test_acc))
    print('-' * 71)

if __name__ == '__main__':
    main()
