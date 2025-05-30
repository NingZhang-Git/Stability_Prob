# Stability of GCNNs and Graph Filters Under Edge Perturbation

This project explores the effects of crafted edge perturbations on graph convolutional neural networks (GCNNs) and various graph filter embeddings.

## Project Structure
```
├── cSBM_visualization.ipynb              # Structural interpretation of cSBM 
├── Experiment on GCNN ebd/               # Experiments on multilayer GCNNs
│   ├── SBM_GCN.ipynb                     # GCN with filter normalized adjacency
│   ├── SBM_GCN_self_loop.ipynb           # GCN with filter normalized adjacency (with self loop)
│   └── SBM_GIN.ipynb                     # GIN
├── Experiments on graph filter ebd/      # Experiments on graph filtering embeddings
│   ├── BA/                               # Barabási–Albert graphs
│   ├── ENZYMES/                          # A real-world biochemical dataset
│   ├── KC/                               # Zachary's karate club
│   ├── SBM/                              # Stochastic Block Models
│   ├── Sensor/                           # Sensor networks
│   └── WS/                               # Watts-Strogatz (small-world) networks           
├── utils/                                # Utility functions
│   └── Perturbe_Algs.py                  # Core edge perturbation algorithms
        ├── random  (Random)              # Randomized baseline algorithm
        ├── pgd-wst (Wst-PGD)             # Projected Gradient Descent optimizing the worst-case embedding perturbation 
        ├── pgd-avg (Prob-PGD)            # Projected Gradient Descent optimizing the expected embedding perturbation 
        
```

## Key Algorithm

**Prob-PGD** is a projected gradient descent method designed to perturb graph structures in a principled way, based on a probabilistic analysis of embedding stability. It is broadly applicable across a variety of GCNN architectures.

The algorithm requires:
- The input **adjacency matrix** of the graph
- A **graph filter function**, which represents the specific graph convolutional operator used by the GCNN

These perturbations can be used to evaluate the **robustness** and **sensitivity** of GCNNs and graph-based embeddings under structural changes.

All Perturbation Algorithms can be found in `utils/Perturbe_Algs.py`.

## Requirements

You can install the required packages via:

```bash
pip install -r requirements.txt
