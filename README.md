# Graph Perturbation and Embedding Experiments

This project explores the effects of crafted edge perturbations on graph neural networks (GNNs) and various graph filter embeddings.

## Project Structure
```
├── cSBM_visualization.ipynb              # Structural interpretation of cSBM 
├── Experiment on GCNN ebd/               # Experiments on multilayer GCNNs
│   ├── SBM_GCN.ipynb                     # GCN
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

```
## 🔍 Purpose

This repository investigates:
- How edge perturbation affects GCNN embedding and downstream classification performance.
- The behavior of different models on synthetic and real-world datasets.

## Key Components

**Prob-PGD** is a probabilistic projected gradient descent method designed to perturb graph structures in a principled way. It is broadly applicable across a variety of Graph Convolutional Neural Network (GCNN) architectures.

The algorithm requires:
- The input **adjacency matrix** of the graph
- A **graph filter function**, which represents the specific graph convolutional operator used by the GCNN

These perturbations can be used to evaluate the **robustness** and **sensitivity** of GCNNs and graph-based embeddings under structural changes.

All Perturbation Algorithms can be found in `utils/Perturbe_Algs.py`.

## Requirements

Recommended:
- Python ≥ 3.7
- PyTorch
- NetworkX
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages via:

```bash
pip install -r requirements.txt
