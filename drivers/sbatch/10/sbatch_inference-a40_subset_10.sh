#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=300G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o train_index-a40-%j.out  # %j = job ID
#SBATCH --constraint="a40|l40s"
#SBATCH --qos=long

nvidia-smi
echo "sbatch_train_index-a40_subset_10.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/drivers
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1


# nohup sh ../commands/train-a40_subset_10.sh | tee train_index-a40_subset_10-$SLURM_JOB_ID.out
 
nohup  sh ../commands/run_ann_data_gen-a40_subset_10.sh  | tee run_ann_data_gen-a40_subset_10-$SLURM_JOB_ID.out


echo "sbatch_train_index-a40_subset_10.sh finished"