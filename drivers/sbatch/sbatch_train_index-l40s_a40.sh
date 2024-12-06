#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o train_index-%j.out  # %j = job ID
#SBATCH --constraint="l40s|a40"
#SBATCH --mail-user=hojaeson@umass.edu

nvidia-smi
echo "subset_run_train_index.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/
export PYTHONPATH=$PYTHONPATH:$(pwd)

nohup python3 -m torch.distributed.launch --master_port=12355 --nproc_per_node=4 run_ann.py \
 --model_type rdot_nll \
 --model_name_or_path  /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/pretrained_bm25  \
 --task_name MSMarco \
 --triplet \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/preprocessed_data \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_top_neg \
 --max_seq_length 512 \
 --per_gpu_train_batch_size 32 \
 --gradient_accumulation_steps 8 \
 --learning_rate 1e-6 \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints_top_neg \
 --warmup_steps 0 \
 --logging_steps 50 \
 --save_steps 500 \
 --single_warmup \
 --top_neg \
 --optimizer lamb  | tee nohup-train-$SLURM_JOB_ID.out &

 
nohup  sh run_ann_data_gen.sh | tee nohup-index-$SLURM_JOB_ID.out &


echo "subset_run_train_index.sh finished"


#

# python -m torch.distributed.launch --nproc_per_node=4 run_ann.py \
#  --model_type rdot_nll \
#  --model_name_or_path /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/pretrained_bm25 \
#  --task_name MSMarco \
#  --triplet \
#  --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/preprocessed_3_data_subset \
#  --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/ann_data \
#  --max_seq_length 512 \
#  --per_gpu_train_batch_size 32 \
#  --gradient_accumulation_steps 2 \
#  --learning_rate 1e-6 \
#  --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/subset_3_outcome/checkpoints \
#  --warmup_steps 5000 \
#  --logging_steps 100 \
#  --save_steps 5000 \
#  --optimizer lamb
 


 
# python3 -m torch.distributed.launch --master_port=12355 --nproc_per_node=4 run_ann.py \
#  --model_type rdot_nll \
#  --model_name_or_path   /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints/checkpoint-400  \
#  --task_name MSMarco \
#  --triplet \
#  --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/preprocessed_data \
#  --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data \
#  --max_seq_length 512 \
#  --per_gpu_train_batch_size 32 \
#  --gradient_accumulation_steps 4 \
#  --learning_rate 1e-5 \
#  --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints \
#  --warmup_steps 0 \
#  --logging_steps 5 \
#  --save_steps 100 \
#  --single_warmup \
#  --optimizer lamb
 