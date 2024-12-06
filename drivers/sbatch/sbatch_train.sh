#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o subset_train-%j.out  # %j = job ID
#SBATCH --constraint="l40s|a40|rtx8000"

nvidia-smi
echo "subset_run_ann.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/drivers
# export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m torch.distributed.launch --nproc_per_node=4 run_ann.py \
 --model_type rdot_nll \
 --model_name_or_path /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_10_outcome/pretrained_bm25 \
 --task_name MSMarco \
 --triplet \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_10_outcome/preprocessed_10_data_subset \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_10_outcome/ann_data \
 --max_seq_length 512 \
 --per_gpu_train_batch_size 32 \
 --gradient_accumulation_steps 2 \
 --learning_rate 1e-6 \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_10_outcome/checkpoints \
 --warmup_steps 5000 \
 --logging_steps 100 \
 --save_steps 5000 \
 --optimizer lamb
 
echo "subset_run_ann.sh finished"




python -m torch.distributed.launch --nproc_per_node=4 run_ann.py \
 --model_type rdot_nll \
 --model_name_or_path /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/pretrained_bm25 \
 --task_name MSMarco \
 --triplet \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/preprocessed_3_data_subset \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/ann_data \
 --max_seq_length 512 \
 --per_gpu_train_batch_size 32 \
 --gradient_accumulation_steps 2 \
 --learning_rate 1e-6 \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/checkpoints \
 --warmup_steps 5000 \
 --logging_steps 100 \
 --save_steps 5000 \
 --optimizer lamb
 