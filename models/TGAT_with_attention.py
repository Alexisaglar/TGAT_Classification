import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import LSTM, Linear

class TGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_nodes, n_classes, heads=8):
        super(TGAT, self).__init__()
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads 

        # GATv2Conv layers to handle spatial dependencies
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=64, heads=8, edge_dim=3)
        self.gat2 = GATv2Conv(in_channels=64*8, out_channels=32, heads=1, edge_dim=3)

        # LSTM layers to handle temporal dependencies
        self.lstm1 = LSTM(input_size=32, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = Linear(128, 1)

        # Fully connected layer for classification
        self.fc = Linear(128, n_classes)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        time_step_outputs = []

        seq_length = X.shape[0]  # Should be 24 in your case
        n_nodes = X.shape[1]     # Should be 33 in your case

        # Process each time step as a separate graph
        for t in range(seq_length):
            X_t = X[t]  # Shape: (n_nodes, in_channels)

            # Apply GAT layers
            X_t = self.gat1(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            X_t = self.gat2(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)

            # Append the processed node features
            time_step_outputs.append(X_t)

        # Stack the outputs to create a sequence
        X = torch.stack(time_step_outputs, dim=0)  # Shape: (seq_length, n_nodes, hidden_dim)
        # Permute to get (n_nodes, seq_length, hidden_dim)
        X = X.permute(1, 0, 2)  # Shape: (n_nodes, seq_length, hidden_dim)

        # Process through LSTM layers
        X, _ = self.lstm1(X)  # X shape: (n_nodes, seq_length, 256)
        X, _ = self.lstm2(X)  # X shape: (n_nodes, seq_length, 128)

        # Apply attention mechanism
        # Reshape X to (n_nodes * seq_length, hidden_size)
        X_reshaped = X.contiguous().view(-1, 128)
        attention_weights = F.softmax(self.attention(X_reshaped), dim=0)  # Shape: (n_nodes * seq_length, 1)
        attention_weights = attention_weights.view(n_nodes, seq_length, 1)  # Shape: (n_nodes, seq_length, 1)

        # Compute context vector as weighted sum of hidden states
        context_vector = torch.sum(X * attention_weights, dim=1)  # Shape: (n_nodes, hidden_size)

        # Pass through the fully connected layer
        X = self.fc(context_vector)  # Shape: (n_nodes, n_classes)

        return X
