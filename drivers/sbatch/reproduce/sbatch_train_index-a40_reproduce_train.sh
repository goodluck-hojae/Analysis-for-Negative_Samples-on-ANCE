#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o sbatch_reproduce_train-a40-%j.out  # %j = job ID
#SBATCH --constraint="a40|l40s"
#SBATCH --qos=long

nvidia-smi
echo "sbatch_train_index-a40_reproduce_train.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/commands
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1

sh run_train_a40_4_gpu_test.sh

echo "sbatch_train_index-a40_reproduce_train.sh finished"