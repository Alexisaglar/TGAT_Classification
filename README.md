# Multi-level TGAT: Temporal Graph Attention Network for Optimal PV Allocation (node classification is in place, working on regressions for 2nd and 3rd level)

Welcome to the **Multi-level TGAT** project! This repository implements a Temporal Graph Attention Network (TGAT) for time-dependent optimal decision tasks. The model leverages both spatial and temporal dependencies to predict class labels, feature prediction for each node in an electrical network.

## Overview

This project focuses on the classification and regression of nodes on the IEEE 33 bus system with time-varying features. The data consists of temporal snapshots of the network, where node connections remain constant, but node features evolve over time.

The TGAT model integrates:
- **Spatial modeling**: Using GATv2Conv layers to capture relationships among nodes.
- **Temporal modeling**: Using GRU layers to learn how node features evolve over time.
- **Classification**: A fully connected layer to predict the class label for each node at the final time step.

The result? 
- A robust model that can classify nodes into one of four distinct categories.

## Features

- Temporal Graph Attention Network architecture combining GATv2Conv and GRU layer.
- Classification of nodes based on evolving features across multiple time steps.
- Supports dynamic input data in the shape `(24, 33, 2)` representing:
  - 24 time steps
  - 33 nodes
  - 2 node features per time step.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alexisaglar/TGAT_Classification.git
   cd TGAT_Classification

2. Install dependencies:
   ```bash
    pip install -r requirements.txt

## Usage

### Training the Model
The training and validation process is managed in `main.py`. Run the script to start training:
    ```bash
    python main.py

### Data
The dataset is loaded from `data/load_classification_100_networks.h5`. Ensure this file is in the correct path, or update the script to point to your dataset location.

### Model Architecture
The TGAT model consists of:
- GATv2Conv layers for learning spatial dependencies.
- GRU layers for capturing temporal patterns.
- A fully connected layer for node classification.

The input data is processed as one graph per time step. The outputs of the GATv2Conv layers are stacked and passed through the LSTMs for final classification.

### Customization
To adapt the model for different node counts, feature dimensions, or time steps, modify the `main.py` script and the model definitions accordingly.

## Results
The model predicts class labels for each node at the final time step, demonstrating its ability to learn spatial and temporal dependencies effectively.
