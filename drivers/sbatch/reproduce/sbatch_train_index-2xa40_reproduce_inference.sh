#!/bin/bash
#SBATCH -c 32  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o sbatch_train_index-2xa40_reproduce_inference-%j.out  # %j = job ID
#SBATCH --constraint="a40|l40s"
#SBATCH --qos=long

nvidia-smi
echo "sbatch_train_index-2xa100_reproduce_inference.sh"
echo "pair with 26342960"
ml load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
conda activate ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/commands
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_BLOCKING_WAIT=1

sh run_ann_data_gen.sh

echo "sbatch_train_index-a100_reproduce_inference.sh finished"