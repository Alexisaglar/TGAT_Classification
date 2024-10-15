import torch
import time
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import LSTM, Linear

class TGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_nodes, seq_length, n_classes, heads=8):
        super(TGAT, self).__init__()
        self.n_nodes = n_nodes
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads 

        # GATv2Conv layers to handle spatial dependencies
        self.gat1 = GATv2Conv(in_channels=2, out_channels=64, heads=8, edge_dim=3)
        self.gat2 = GATv2Conv(in_channels=64*8, out_channels=512, heads=1, edge_dim=3)

        # LSTM layers to handle temporal dependencies
        # self.lstm1 = LSTM(input_size=hidden_channels*self.n_nodes, hidden_size=128, batch_first=True)
        self.lstm1 = LSTM(input_size=512*33, hidden_size=128, batch_first=True)
        self.lstm2 = LSTM(input_size=128, hidden_size=64, batch_first=True)

        # Fully connected layer for classification/regression
        self.fc = Linear(64, n_classes)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # List to hold the processed outputs for each time step
        time_step_outputs = []

        # Process each time step as a separate graph
        for t in range(self.seq_length):
            X_t = X[t]  # Get features for time step t (shape: [n_nodes, in_channels])

            # Apply GAT layers (spatial attention)
            X_t = self.gat1(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            X_t = self.gat2(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            print(X_t, X_t.shape)

            # Append the processed node features for this time step
            time_step_outputs.append(X_t)

        # Stack the outputs to create a sequence (shape: [batch_size, seq_length, n_nodes * hidden_channels])
        # print(f"X.shape after GAT layers: {X.shape}")

        X = torch.stack(time_step_outputs, dim=0)  # Shape: (seq_length, n_nodes, hidden_dim)
        print(f"X stack shape: {X.shape}")
        X = X.permute(1, 0, 2)  # Reshape to (batch_size, seq_length, n_nodes * hidden_channels)
        print(f"X.shape after permutation: {X.shape}")

        # Reshape for LSTM input (ensure this matches input_size of the LSTM)
        # X = X.reshape(-1, self.seq_length, self.n_nodes * self.hidden_channels)  # (batch_size, seq_length, n_nodes * hidden_dim)
        X = X.reshape(1, self.seq_length, self.n_nodes * self.hidden_channels * self.heads)  # (batch_size, seq_length, features)
        print(f"X.shape before LSTM: {X.shape}")

        # Temporal attention through LSTMs
        X, _ = self.lstm1(X)
        print(f"X.shape after LSTM1: {X.shape}")
        X, _ = self.lstm2(X)
        print(f"X.shape after LSTM2: {X.shape}")

        # Use the last output of the LSTM for final classification

        X = X[:, -1, :]  # Shape: (batch_size, hidden_size) -> last time step
        print(f"X.shape last time step: {X.shape}")
        # X = X.view(33, self.n_nodes, -1)  # (batch_size, n_nodes, hidden_size)
        print(f"X.shape after taking last time step: {X.shape}")

        # Now pass it through the fully connected layer
        X = self.fc(X)  # Shape: (batch_size, n_nodes, n_classes)
        print(f"X.shape after FC layer: {X.shape}")

        return X
