# Minimum Vertex Cover with Physics-Inspired Graph Neural Networks

This project adapts a physics-inspired Graph Neural Network (GNN) framework to solve the **Minimum Vertex Cover (MVC)** problem. The original GNN model was designed for the Maximum Independent Set (MIS) problem, and has been modified to address MVC using QUBO-based loss formulation and post-processing to ensure feasibility.

## Overview

- **Original Problem:** Maximum Independent Set (MIS)
- **Adapted Problem:** Minimum Vertex Cover (MVC)
- **Tech Stack:** PyTorch, DGL, SHARCNET, Python
- **Evaluation:** Regular 3- and 5-degree synthetic graphs, up to 100,000 nodes
- **Baselines:** ILP via CPLEX, greedy heuristics, and approximation algorithms

## Method

I adapted the physics-inspired GNN approach introduced by Amazon Science to minimize a QUBO-formulated objective function using unsupervised training. After generating soft node assignments, I apply a post-classification repair step to project outputs into valid vertex covers.

Key features:
- Use of a QUBO matrix to encode the MVC objective
- GNN architecture trained via gradient descent on relaxed binary variables
- Greedy projection used to enforce cover feasibility after training
- Comparison against ILP, approximation, and heuristic baselines

## Files

- `gnn_example.py` – Modified training and testing pipeline for MVC
- `utils.py` – Unmodified utility file from the original Amazon repository
- `requirements.txt` – Environment file combining custom and inherited packages
- `LICENSE-SAMPLECODE.txt` – MIT-0 license for reused code
- `LICENSE-DOCS.txt` – Creative Commons license for reused documentation
- `LICENSE-SUMMARY.txt` – Official license summary provided by Amazon outlining the dual licensing of their code and documentation  

## Environment Setup

This project uses Python 3.8 and includes packages from `conda-forge`, `dglteam`, and `pytorch` channels. The following command is recommended to create the environment:

```bash
conda create -n <environment_name> python=3.8 --file requirements.txt -c conda-forge -c dglteam -c pytorch
```

> **Note:** This setup is tested with Python 3.8.5 and 3.8.12. Newer Python versions may lead to package conflicts.

After activating the environment, you can run the main script directly:

```bash
conda activate <environment_name>
python gnn_example.py
```

### CPLEX Installation

> **Note:** CPLEX must be installed manually. It is not included in the `requirements.txt` file and should be installed separately through IBM.


## Attribution

This project builds upon code from the [Amazon Science repository](https://github.com/amazon-science/co-with-gnns-example) related to the paper:

> Schuetz, M. J. A., Brubaker, J. K., & Katzgraber, H. G.  
> _Combinatorial Optimization with Physics-Inspired Graph Neural Networks_  
> arXiv:2107.01188 – [View on arXiv](https://arxiv.org/abs/2107.01188)

- `utils.py` is reused as-is.
- `gnn_example.py` was adapted to solve the Minimum Vertex Cover problem.
- The `requirements.txt` is partially inherited and adjusted for this implementation.

## License Summary

- The reused code from Amazon is licensed under the **MIT-0 license**. See `LICENSE-SAMPLECODE.txt`.
- The documentation reused from Amazon is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License**. See `LICENSE-DOCS.txt`.
- A summary of these license terms is provided in `LICENSE-SUMMARY.txt`.
- The rest of this project (code and documentation) is not licensed for reuse. This repository is shared for academic and personal learning purposes only.

## Citation

If you use this repository or build upon it, please cite the original work:

```bibtex
@article{Schuetz2021,
  title={Combinatorial Optimization with Physics-Inspired Graph Neural Networks},
  author={Schuetz, Martin J A and Brubaker, J Kyle and Katzgraber, Helmut G},
  journal={arXiv preprint arXiv:2107.01188},
  year={2021}
}
```
