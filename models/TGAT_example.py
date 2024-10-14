import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import LSTM, Linear

class TemporalGATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_nodes, seq_length, n_classes, heads=8):
        super(TemporalGATNet, self).__init__()
        self.n_nodes = n_nodes
        self.seq_length = seq_length
        self.in_channels = in_channels

        # GATv2Conv layers to handle spatial dependencies
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, edge_dim=3)
        self.gat2 = GATv2Conv(in_channels=hidden_channels*heads, out_channels=hidden_channels, heads=heads, edge_dim=3)

        # LSTM layers to handle temporal dependencies
        self.lstm1 = LSTM(input_size=hidden_channels*self.n_nodes, hidden_size=128, batch_first=True)
        self.lstm2 = LSTM(input_size=128, hidden_size=64, batch_first=True)

        # Fully connected layer for classification/regression
        self.fc = Linear(64, n_classes)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Pass the input through two GAT layers (spatial attention)
        X = self.gat1(X, edge_index, edge_attr)
        X = F.elu(X)

        X = self.gat2(X, edge_index, edge_attr)
        X = F.elu(X)

        # Reshape the output for LSTM input:
        # Reshape to (batch_size, seq_length, n_nodes * hidden_channels)
        batch_size = data.num_graphs
        X = X.view(batch_size, self.seq_length, self.n_nodes * X.size(-1))  # (B, T, N*hidden_dim)

        # Temporal attention through LSTMs
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)

        # Use the last output of the LSTM for final classification
        X = X[:, -1, :]  # Take the output of the last time step

        # Final classification layer
        X = self.fc(X)

        return X
