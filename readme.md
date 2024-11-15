
# TGAT Classification: Temporal Graph Attention Network for Node Classification

Welcome to the **TGAT Classification** project! This repository implements a Temporal Graph Attention Network (TGAT) for time-dependent node classification tasks. The model leverages both spatial and temporal dependencies to predict class labels for each node in a dynamic network.

## Overview

This project focuses on the classification of nodes in a 33-node network system with time-varying features. The data consists of temporal snapshots of the network, where node connections remain constant, but node features evolve over time.

The TGAT model integrates:
- **Spatial modeling**: Using GATv2Conv layers to capture relationships among nodes.
- **Temporal modeling**: Using LSTM layers to learn how node features evolve over time.
- **Classification**: A fully connected layer to predict the class label for each node at the final time step.

The result? A robust model that can classify nodes into one of four distinct categories.

## Features

- Temporal Graph Attention Network architecture combining GATv2Conv and LSTM layers.
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
