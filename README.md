# Impact Analysis of Hard Negative Sample Rankings in ANCE, 646 Project, Fall 2024
Hojae Son*, Deepesh Suranjandass*, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk

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
