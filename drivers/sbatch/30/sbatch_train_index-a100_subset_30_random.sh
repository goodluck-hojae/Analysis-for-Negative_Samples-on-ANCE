#!/bin/bash
#SBATCH -c 64  # Number of Cores per Task
#SBATCH --mem=300G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o train_index-a100-%j.out  # %j = job ID
#SBATCH --constraint="a100"
#SBATCH --qos=long


nvidia-smi
echo "sbatch_train_index-a100_subset_30_random.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/drivers
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1

sh ../commands/subset_30_train-a100_random.sh | tee subset_30_train-a100-$SLURM_JOB_ID.out &
 
sh ../commands/subset_30_run_ann_data_gen-a100_random.sh  | tee subset_30_run_ann_data_gen-a100-$SLURM_JOB_ID.out


echo "sbatch_train_index-a100_subset_30_random.sh finished"