export CUDA_VISIBLE_DEVICES=6,7
dt=`date '+%Y%m%d_%H%M%S'`

dataset="csqa"
model='roberta-large'
#shift
#shift
args=$@


# echo "***** hyperparameters *****"
# echo "dataset: $dataset"
# echo "enc_name: $model"
# echo "batch_size: $bs"
# echo "learning_rate: elr $elr dlr $dlr"
# echo "edge_encoder_dim $enc_dim gsc_layer $k"
# echo "******************************"

save_dir_pref='experiments'
logs_dir_pref='logs/csqa'
mkdir -p $save_dir_pref
mkdir -p $logs_dir_pref


n_epochs=30
bs=128
mbs=4
ebs=8
enc_dim=32

elr="2e-5"
dlr="1e-4"
weight_decay="1e-2"
dropout="1e-1"
dropoutf="1e-1"
drop_ratio="0.05"
k=2

tr_dim=1024
ffn_dim=2048
num_heads=16
lambda="10"


###### Training ######
for seed in 0; do
  python3 -u main_qat.py --dataset $dataset \
      --encoder $model -k $k \
      -elr $elr -dlr $dlr -bs $bs -mbs ${mbs} -ebs ${ebs} --weight_decay ${weight_decay} --seed $seed \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.ori2.metapath.2.q2a.seq.pk \
      --dev_adj data/${dataset}/graph/dev.graph.adj.ori2.metapath.2.q2a.seq.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.ori2.metapath.2.q2a.seq.pk\
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --max_seq_len 88     \
      --num_relation 38    \
      --unfreeze_epoch 4 \
      --log_interval 10 \
      --transformer_dim ${tr_dim} \
      --transformer_ffn_dim ${ffn_dim} \
      --num_heads ${num_heads} \
      --dropouttr ${dropout} \
      --dropoutf ${dropoutf} \
      --lr_schedule "warmup_linear" \
      --save_model \
      --max_node_num 44 \
      --inverse_relation \
      --drop_ratio ${drop_ratio} \
      --lambda_rpe ${lambda} \
  | tee -a $logs_dir_pref/newFT_path.${dataset}_${elr}.${dlr}.${weight_decay}.${dropout}_${tr_dim}.${ffn_dim}.${num_heads}__seed${seed}_${dt}.log.txt
done
