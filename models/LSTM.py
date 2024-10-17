
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNodeClassifierWithAttention(nn.Module):
    def __init__(self, in_channels, hidden_size, n_classes):
        super(LSTMNodeClassifierWithAttention, self).__init__()
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        # X shape: (n_nodes, seq_length, in_channels)
        X, _ = self.lstm(X)  # X shape: (n_nodes, seq_length, hidden_size)

        # Compute attention weights
        attention_scores = self.attention(X)  # Shape: (n_nodes, seq_length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over seq_length

        # Compute context vector
        context_vector = torch.sum(X * attention_weights, dim=1)  # Shape: (n_nodes, hidden_size)

        # Pass through the fully connected layer
        X = self.fc(context_vector)  # Shape: (n_nodes, n_classes)

        return X
