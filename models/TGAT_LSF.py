import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import GRU, Linear

class TGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8):
        super(TGAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads

        # GAT layers for spatial dependencies
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

        # GRU for temporal dependencies per node
        self.gru = GRU(
            input_size=hidden_channels,
            hidden_size=256,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer for graph-level scalar regression
        self.fc = Linear(256 * 2, 33)  

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # X shape: (24, 33, 2) => (seq_length, n_nodes, in_channels)
        # print(X.shape)
        seq_length, n_nodes, _ = X.shape

        # List to collect node embeddings at each time step
        node_embeddings = []

        for t in range(seq_length):
            X_t = X[t]  # Shape: (33, 2)

            # Apply GAT layers
            X_t = self.gat1(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            X_t = self.gat2(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)

            node_embeddings.append(X_t)  # Shape of X_t: (33, hidden_channels)

        # Stack node embeddings over time
        # node_embeddings shape: (seq_length, n_nodes, hidden_channels)
        node_embeddings = torch.stack(node_embeddings, dim=0)

        # Permute to get (n_nodes, seq_length, hidden_channels)
        node_embeddings = node_embeddings.permute(1, 0, 2)  # Shape: (33, 24, hidden_channels)

        # Process temporal dependencies per node using GRU
        output, h_n = self.gru(node_embeddings)  # output shape: (33, 24, 256), h_n shape: (2, 33, 128)

        # Concatenate the final hidden states from both directions
        # h_n shape: (num_directions, batch_size, hidden_size)
        h_n_forward = h_n[0]  # Shape: (33, 128)
        h_n_backward = h_n[1]  # Shape: (33, 128)
        h_n_combined = torch.cat((h_n_forward, h_n_backward), dim=1)  # Shape: (33, 256)

        # Global mean pooling to get graph-level embedding
        graph_embedding = torch.mean(h_n_combined, dim=0, keepdim=True)  # Shape: (1, 256)

        # Pass through the fully connected layer for graph-level regression
        out = self.fc(graph_embedding)  # Shape: (1, 1)

        return out
