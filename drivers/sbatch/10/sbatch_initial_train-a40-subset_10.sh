#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o initial_train_subset-a40-%j.out  # %j = job ID
#SBATCH --constraint="l40s|a40"

nvidia-smi
echo "sbatch_initial_train_subset_10.sh"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/
export PYTHONPATH=$PYTHONPATH:$(pwd)

sh commands/run_ann_data_gen_initial-a40_subset_10.sh
echo "commands/sbatch_initial_train_subset_10.sh"

