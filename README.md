# Graph Perturbation and Embedding Experiments

This project explores the effects of edge perturbation on graph neural networks (GNNs) and various graph filter embeddings. The central algorithms for perturbing graph edges are implemented in the `utils/Perturbe_Algs.py` module.

## Project Structure
'''
├── cSBM_visualization.ipynb              # Visualization of SBM results
├── Experiment on GCNN ebd/               # Experiments on Graph Convolutional Neural Networks
│   ├── SBM_GCN.ipynb                     # GCN experiments on SBM graphs
│   └── SBM_GIN.ipynb                     # GIN experiments on SBM graphs
├── Experiments on graph filter ebd/      # Experiments on graph filtering-based embeddings
│   ├── BA/                               # Experiments on Barabási–Albert graphs
│   ├── ENZYMES/                          # Experiments on biochemical datasets
│   ├── KC/                               # KC graph datasets
│   ├── SBM/                              # Stochastic Block Model graphs
│   ├── Sensor/                           # Sensor graph datasets
│   └── WS/                               # Watts-Strogatz small-world networks
├── README.md                             # Project documentation
├── utils/                                # Utility functions
│   ├── __pycache__/                      # Compiled Python files
│   └── Perturbe_Algs.py                  # Core edge perturbation algorithms

'''
## 🔍 Purpose

This repository investigates:
- How edge perturbation affects GNN performance.
- The robustness of graph filter-based embeddings.
- The behavior of different models on synthetic and real-world datasets.

## Key Components

### Perturbation Algorithms

**Location:** `utils/Perturbe_Algs.py`

This module contains implementations for algorithms that randomly or strategically perturb the edges of graphs. These perturbations simulate noise or adversarial attacks and are used to test model robustness.

### GCNN Experiments

Notebooks in `Experiment on GCNN ebd/` demonstrate how Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks (GINs) react to perturbed graph structures.

### Graph Filter Embeddings

Contained in `Experiments on graph filter ebd/`, these experiments assess how perturbations influence embedding methods like spectral filters or diffusion-based embeddings.

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
