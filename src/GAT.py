import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):
    """
    GAT for node classification using PyTorch Geometric
    """

    def __init__(
        self, in_features: int, out_features: int, n_heads: int, dropout: float = 0
    ):
        """
        Initialize the GAT model with parameters.
        :param in_features: Node input features
        :param out_features: Node output features
        :param n_heads: Number of attention heads
        :param dropout: Dropout probability
        """
        super(GAT, self).__init__()
        self.gat_conv = GATv2Conv(
            in_channels = in_features,
            out_channels = out_features // n_heads,
            heads=n_heads,
            edge_dim = edge_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor)
        """
        Forward pass of GATv2 with edge attributes.
        :param x: Input node features, shape [n_nodes, in_features]
        :param edge_index: Graph edge indices, shape [2, num_edges]
        :param edge_attr: Edge attributes, shape [num_edges, edge_dim]
        :return: Output node embeddings
        """
        x = self.gatv2_conv(x, edge_index, edge_attr)

        return x
