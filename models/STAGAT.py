import torch
import torch.nn as nn
import torch.nn.functional as F

class RAU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RAU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Learnable parameters for attention
        self.attention = nn.Linear(input_size + hidden_size, 1)
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial hidden state

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # Shape: (batch_size, input_size)

            # Compute attention weights
            att_input = torch.cat([x_t, h_t], dim=1)  # Shape: (batch_size, input_size + hidden_size)
            att_score = F.softmax(self.attention(att_input), dim=0)  # Shape: (batch_size, 1)

            # Weighted input
            x_t_att = att_score * x_t  # Apply attention to the input

            # Update hidden state using GRUCell
            h_t = self.rnn_cell(x_t_att, h_t)  # Shape: (batch_size, hidden_size)

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, hidden_size)
        return outputs, h_t  # Return all hidden states and the final hidden state

class TGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, heads=8):
        super(TGAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
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

        # RAU for temporal dependencies per node
        self.rau = RAU(
            input_size=hidden_channels,
            hidden_size=256,
            num_layers=4  # RAU handles multiple layers internally
        )

        # Fully connected layer for per-node classification
        self.fc = Linear(256, n_classes)  # RAU outputs final hidden size

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        seq_length, n_nodes, _ = X.shape

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
        node_embeddings = torch.stack(node_embeddings, dim=0)

        # Permute to get (n_nodes, seq_length, hidden_channels)
        node_embeddings = node_embeddings.permute(1, 0, 2)  # Shape: (33, 24, hidden_channels)

        # Process temporal dependencies using RAU
        output, h_n = self.rau(node_embeddings)  # output shape: (33, 24, 256)

        # Take the last hidden state for classification
        out = self.fc(h_n)  # Shape: (33, n_classes)

        return out
