#!/bin/bash
#SBATCH -c 64  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1 # Number of GPUs
#SBATCH --nodes=1
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o train-initial-%j.out  # %j = job ID
#SBATCH --constraint=a100

nvidia-smi
echo "run_ann_data_gen.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1

python -m torch.distributed.launch --nproc_per_node=1 drivers/run_ann_data_gen.py \
 --training_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints \
 --init_model_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/pretrained_bm25 \
 --model_type rdot_nll \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_neg_initial \
 --cache_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/cache \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/preprocessed_data \
 --max_seq_length 512 \
 --per_gpu_eval_batch_size 2048 \
 --topk_training 1000 \
 --negative_sample 200 \
 --server_port 12345 \
 --end_output_num -1 \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_neg_initial


echo "run_ann_data_gen.sh finished"




# python -m torch.distributed.launch --nproc_per_node=1 drivers/run_ann_data_gen.py \
#  --training_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints \
#  --init_model_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/pretrained_bm25 \
#  --model_type rdot_nll \
#  --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_bottom_neg \
#  --cache_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/cache \
#  --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/preprocessed_data \
#  --max_seq_length 512 \
#  --per_gpu_eval_batch_size 2048 \
#  --topk_training 1000 \
#  --negative_sample 200 \
#  --server_port 12345 \
#  --end_output_num -1 \
#  --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_neg_initial
