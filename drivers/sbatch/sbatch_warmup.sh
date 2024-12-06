#!/bin/bash
#SBATCH -c 64  # Number of Cores per Task
#SBATCH --mem=150G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2 # Number of GPUs
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o warmup-%j.out  # %j = job ID
#SBATCH --constraint=a100

nvidia-smi
echo "sbatch_warmup.sh"
conda activate  /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/ance
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd /home/hojaeson_umass_edu/hojae_workspace/vector_database/ANCE/commands
sh run_train_warmup.sh 

echo "sbatch_warmup.sh finished"
