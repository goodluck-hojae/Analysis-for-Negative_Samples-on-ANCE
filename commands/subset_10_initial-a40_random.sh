#!/bin/bash
#
# This script is for generate ann data for a model in training
#
# For the overall design of the ann driver, check run_train.sh
#
# This script continuously generate ann data using latest model from model_dir
# For training, run this script after initial ann data is created from run_train.sh
# Make sure parameter used here is consistent with the training script

# # Passage ANCE(FirstP) 
gpu_no=4
seq_length=512
model_type=rdot_nll
tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
base_data_dir="/datasets/ai/msmarco/passage/"

# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
preprocessed_data_dir="/home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/preprocessed_data"

job_name="OSPass512"


# # Document ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# model_type=rdot_nll
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc512"

# # Document ANCE(MaxP)
# gpu_no=1
# seq_length=2048
# model_type=rdot_nll_multi_chunk
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc2048"

##################################### Inital ANN Data generation ################################
# model_dir="${base_data_dir}${job_name}/"
model_dir="/home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/checkpoints_4_random"
model_ann_data_dir="/home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/ann_data_4_random"

mkdir -p ${model_dir}
mkdir -p ${model_ann_data_dir}

# pretrained_checkpoint_dir="warmup checkpoint path"
pretrained_checkpoint_dir="/home/hojaeson_umass_edu/hojae_workspace/shared/ance/pretrained_bm25-150000"

MASTER_PORT=29501


export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1

initial_data_gen_cmd="\
python -m torch.distributed.launch  --master_port=29502 --nproc_per_node=$gpu_no drivers/run_ann_data_gen.py \
 --training_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/checkpoints_random \
 --init_model_dir ${pretrained_checkpoint_dir} \
 --model_type rdot_nll \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/ann_data_4_random \
 --cache_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/cache \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/preprocessed_data \
 --max_seq_length 512 \
 --per_gpu_eval_batch_size 512 \
 --topk_training 200 \
 --negative_sample 20 \
 --server_port 12345 \
 --end_output_num -1 \
 --ann_chunk_factor 20 \
 --bottom_neg true \
 --ann_measure_topk_mrr false \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/data/10/ann_data_4_random"
 
echo $initial_data_gen_cmd
eval $initial_data_gen_cmd