#!/bin/bash
#SBATCH -c 64  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o train_index-a100-%j.out  # %j = job ID
#SBATCH --constraint="a100"
#SBATCH --qos=long

nvidia-smi
echo "subset_run_train_index_bottom_neg.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/drivers
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1


nohup python3 -m torch.distributed.launch --master_port=12355 --nproc_per_node=2 run_ann.py \
 --model_type rdot_nll \
 --model_name_or_path  /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/pretrained_bm25  \
 --task_name MSMarco \
 --triplet \
 --data_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/preprocessed_data \
 --ann_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/ann_data_bottom_neg \
 --max_seq_length 512 \
 --per_gpu_train_batch_size 30 \
 --gradient_accumulation_steps 8 \
 --learning_rate 1e-6 \
 --output_dir /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/outcome/checkpoints_bottom_neg \
 --warmup_steps 0 \
 --logging_steps 50 \
 --save_steps 500 \
 --single_warmup \
 --optimizer lamb  | tee nohup-train-$SLURM_JOB_ID.out &
 
 
nohup  sh run_ann_data_gen-a100_bottom_neg.sh  | tee nohup-index-a100-$SLURM_JOB_ID.out


echo "subset_run_train_index_bottom_neg.sh finished"