#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o initial_train-%j.out  # %j = job ID
#SBATCH --constraint="l40s|a40|rtx8000"

nvidia-smi
echo "drivers/sbatch_initial_train_subset.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/drivers
# export PYTHONPATH=$PYTHONPATH:$(pwd)
MASTER_ADDR=localhost MASTER_PORT=12355  python -m torch.distributed.launch --master_port=12355 --nproc_per_node=4 run_ann_data_gen.py \
 --training_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/checkpoints \
 --init_model_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/pretrained_bm25 \
 --model_type rdot_nll \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/ann_data \
 --cache_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/cache \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/preprocessed_3_data_subset \
 --max_seq_length 512 \
 --per_gpu_eval_batch_size 2048 \
 --topk_training 200 \
 --negative_sample 20 \
 --server_port 12345 \
 --end_output_num -1

 
echo "drivers/sbatch_initial_train_subset.sh"

