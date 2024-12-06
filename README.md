# Impact Analysis of Hard Negative Sample Rankings in ANCE, 646 Project, Fall 2024
Hojae Son*, Deepesh Suranjandass*

This repo is inspired from the ANCE codebase [https://github.com/microsoft/ANCE/] & paper [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf) 

We investigate the impact of hard negative sampling strategies
in the ANCE [11] (Approximate Nearest Neighbor Negative Contrastive
Learning) framework by analyzing how the ranking positions
of negative samples affect model convergence and generalization.
Our hypothesis suggests that the degree of negative sample
hardness significantly influences training dynamics and model performance.
Using a subset of MS MARCO, we conduct experiments
comparing different ranking segments for negative sampling to
understand their impact on training efficiency and model effectiveness.

# Reproducing Results with 10% subset data

### To download raw dataset, please refer commands/data_download.sh script
### To create subset, please refer data/create_subset.py 
The logs are located in `results` directory with the following files
- `warmup-26334398.out`
- `subset_10_train-a40-26398175-ann_data_4.out`
- `subset_10_train-a40-26398212-ann_data_4_bottom_neg.out`
- `subset_10_train-a40-26449628-ann_data_4_random.out`
- `subset_10_train-a40-26456768-ann_data_4_bottom_neg_only.out`

To reproduce the results in the `results` directory, execute the provided commands. The experiments were conducted using a SLURM cluster, so ensure your SLURM settings are configured correctly.

## Commands
Run the following scripts for the experiments:
- `commands/run_train_warmup.sh`
- `commands/subset_10_train-a40.sh`
- `commands/subset_10_train-a40_bottom_neg.sh`
- `commands/subset_10_train-a40_bottom_neg_only.sh`
- `commands/subset_10_train-a40_random.sh`

![warm_up_mrr](https://github.com/user-attachments/assets/08cd1ae1-4a04-4a61-95e9-af9a7ea9ae6d)
![ndcg_comparison](https://github.com/user-attachments/assets/87250c8c-504b-4c6e-9617-c9ffb5b99ac3)


## SLURM Configurations
Use these SLURM batch scripts to submit jobs for different experiment setups:

- `drivers/10/sbatch_train_index-a40_subset_10.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_bottom_neg.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_bottom_neg_only.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_random.sh`
- `drivers/sbatch_warmup.sh`

