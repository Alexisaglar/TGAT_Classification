import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import GRU, Linear
from torch.nn import LSTM

class TGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, heads=8):
        super(TGAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.heads = heads

        # GAT layers for spatial dependencies remain the same
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=3
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            edge_dim=3
        )

        # Replace GRU with LSTM
        self.lstm = LSTM(
            input_size=hidden_channels,
            hidden_size=256,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer for per-node classification
        self.fc = Linear(256 * 2, n_classes)  # Multiply by 2 for bidirectional LSTM

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        seq_length, n_nodes, _ = X.shape

        # List to collect node embeddings at each time step
        node_embeddings = []

        for t in range(seq_length):
            X_t = X[t]  # Shape: (n_nodes, in_channels)

            # Apply GAT layers
            X_t = self.gat1(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            X_t = self.gat2(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)

            node_embeddings.append(X_t)  # Shape of X_t: (n_nodes, hidden_channels)

        # Stack node embeddings over time
        node_embeddings = torch.stack(node_embeddings, dim=0)
        # Permute to get (n_nodes, seq_length, hidden_channels)
        node_embeddings = node_embeddings.permute(1, 0, 2)

        # Process temporal dependencies per node using LSTM
        output, (h_n, c_n) = self.lstm(node_embeddings)

        # Extract the final hidden states from the last layer for both directions
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # Since num_layers=4 and bidirectional=True, num_directions=2
        # Therefore, h_n shape: (8, n_nodes, hidden_size)
        # Get the last layer's hidden states
        h_n_forward = h_n[-2]  # Shape: (n_nodes, hidden_size)
        h_n_backward = h_n[-1]  # Shape: (n_nodes, hidden_size)
        h_n_combined = torch.cat((h_n_forward, h_n_backward), dim=1)  # Shape: (n_nodes, hidden_size * 2)

        # Pass through the fully connected layer for per-node classification
        out = self.fc(h_n_combined)  # Shape: (n_nodes, n_classes)

        return out
