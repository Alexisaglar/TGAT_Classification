import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GraphAttentionLayer(torch.nn.Module):
    """
    GAT for Node classification
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
    ):
        """
        initialize model with parameters
        :param in_features --> nodes input features
        :param out_features --> nodes output features
        :param n_heads --> number of attention heads
        :param is_concat --> boolean for layer concatenation or averaging
        :param dropout --> dropout probability on connected layers
        """
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            # Calculate the number of dimension per head for concatenation
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            # For averaging attention
            self.n_hidden = out_features

        # Linear layer for initial transformation, generate node embeddings
        self.linear = torch.nn.Linear(in_features, self.n_hidden, *n_heads, bias=False)
        # Linear layer to compute attention score e_ij
        self.attn = torch.nn.Linear(self.n_hidden * 2, 1, bias=False)
        # Activation attention score e_ij
        self.activation = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention a_ij
        self.softmax = torch.nn.Softmax(dim=1)
        # Dropout layer to be applied
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        param: H --> is the input node embeddings of shape [n_nodes, in_features].
        param: adj_mat --> is the adjacency matrix of shape [n_nodes, n_nodes, n_heads],
                            as adjacency matrix is the same for all heads = [n_nodes, n_nodes, 1]
        """
        # number of nodes
        n_nodes = h.shape[0]
        # The initial transformation for each head k: g_i^k = W^k h_i
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        # g repeat calculates g_i||g_j for all pairs repeated n_nodes times
        g_repeat = g.repeat(n_nodes, 1, 1)

        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        e = self.activation(self.attn(g_concat))

        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)

        else:
            return attn_res.mean(dim=1)

