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
gpu_no=2
seq_length=512
model_type=rdot_nll
tokenizer_type="roberta-base"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
preprocessed_data_dir="/home/hojaeson_umass_edu/hojae_workspace/project/ance/data/30/preprocessed_data"

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
model_dir="/home/hojaeson_umass_edu/hojae_workspace/project/ance/data/30/checkpoints_2_random"
model_ann_data_dir="/home/hojaeson_umass_edu/hojae_workspace/project/ance/data/30/ann_data_2_random"

mkdir -p ${model_dir}
mkdir -p ${model_ann_data_dir}

# pretrained_checkpoint_dir="warmup checkpoint path"
pretrained_checkpoint_dir="/home/hojaeson_umass_edu/hojae_workspace/shared/ance/pretrained_bm25-150000"

MASTER_PORT=29501
initial_data_gen_cmd="\
python -m torch.distributed.launch --master_port=29501 --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size 512 --topk_training 200 --negative_sample 20 --ann_chunk_factor 20 --ann_measure_topk_mrr false \
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd
