# Impact Analysis of Hard Negative Sample Rankings in ANCE, 646 Project, Fall 2024
Hojae Son*, Deepesh Suranjandass*

This repo is inspired from 
- ANCE codebase [https://github.com/microsoft/ANCE/] 
- ANCE paper [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf) 

We investigate the impact of hard negative sampling strategies
in the ANCE [11] (Approximate Nearest Neighbor Negative Contrastive
Learning) framework by analyzing how the ranking positions
of negative samples affect model convergence and generalization.
Our hypothesis suggests that the degree of negative sample
hardness significantly influences training dynamics and model performance.
Using a subset of MS MARCO, we conduct experiments
comparing different ranking segments for negative sampling to
understand their impact on training efficiency and model effectiveness.

## Code Modifications

Our primary code changes are in `drivers/run_ann_data_gen.py`. Below is a summary of the key logic modifications:

### Select Top K Samples
The changes focus on how positive and negative samples are selected during data generation
The following code block is a part of our modifications:

```python
if SelectTopK:
    selected_ann_idx = list(top_ann_pid[:args.negative_sample + 1])
    count_negative_sample = args.negative_sample
    
    # Add bottom negative samples if enabled
    if args.bottom_neg:                             
        selected_ann_idx.extend(top_ann_pid[-args.negative_sample:])
        count_negative_sample = 2 * args.negative_sample
        # print('count_negative_sample = 2 * args.negative_sample')
    
    # Only use bottom negative samples if both flags are set
    if args.bottom_neg and args.bottom_only:
        selected_ann_idx = list(top_ann_pid[-args.negative_sample:])
        count_negative_sample = 2 * args.negative_sample
        # print('count_negative_sample = args.negative_sample')
```


# Reproducing Results with 10% subset data
### To download raw dataset, please refer commands/data_download.sh script
### To create subset, please refer data/create_subset.py 
The logs are located in `results` directory with the following files
- `warmup-26334398.out`
- `subset_10_train-a40-26398175-ann_data_4.out`
- `subset_10_train-a40-26398212-ann_data_4_bottom_neg.out`
- `subset_10_train-a40-26449628-ann_data_4_random.out`
- `subset_10_train-a40-26456768-ann_data_4_bottom_neg_only.out`

## Commands
Run the following scripts for the experiments:
- `commands/run_train_warmup.sh`
- `commands/subset_10_train-a40.sh`
- `commands/subset_10_train-a40_bottom_neg.sh`
- `commands/subset_10_train-a40_bottom_neg_only.sh`
- `commands/subset_10_train-a40_random.sh`

### Warm-Up MRR
<img src="https://github.com/user-attachments/assets/08cd1ae1-4a04-4a61-95e9-af9a7ea9ae6d" alt="warm_up_mrr" width="500">

### NDCG Comparison
<img src="https://github.com/user-attachments/assets/87250c8c-504b-4c6e-9617-c9ffb5b99ac3" alt="ndcg_comparison" width="500">

## SLURM Configurations
To reproduce the results in the `results` directory, execute the provided commands. 
The experiments were conducted using a SLURM cluster, so please refer the hardware settings and configurations
- `drivers/10/sbatch_train_index-a40_subset_10.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_bottom_neg.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_bottom_neg_only.sh`
- `drivers/10/sbatch_train_index-a40_subset_10_random.sh`
- `drivers/sbatch_warmup.sh`

## Report
Please refer the details, CS646_Final_Project.pdf


## Contact
For any questions or further information, please contact:
- Hojae Son: hojaeson@umass.edu
- Deepesh Suranjandass: dsuranjandas@umass.edu
