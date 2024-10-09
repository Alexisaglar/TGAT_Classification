import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import lstm


class TGAT(torch.nn.Module):
    """
    Temporal GAT for Node classification
    """

    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0.0):
        """
        initialize model with parameters
        :param in_channels --> dimension of input channels
        :param out_channels --> dimension of output channels
        :param n_nodes --> nodes in the graph
        :param heads --> number of multi-head attention layers
        :param heads --> number of multi-head attention layers
        :param dropout --> dropout probability on connected layers
        """
        super(TGAT, self).__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        # self.n_preds = 9

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            dropout=0,
            concat=False,
        )

        ### add two lstm layers
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        def init_lstm(input_size, hidden_size):
            lstm = torch.nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=1
            )
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    torch.nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
            return lstm

        # Initialize two LSTM layers
        self.lstm1 = init_lstm(input_size=self.n_nodes, hidden_size=lstm1_hidden_size)
        self.lstm2 = init_lstm(input_size=self.n_nodes, hidden_size=lstm2_hidden_size)

    def forward(self, data, device):
        """
        Forward pass for temporal GAT
        :param data data to make a pass on
        :param device device it is operating on
        """
        x, edge_index = data.x, data.edge_index
        # apply dropout
        if device == "cpu":
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)

        # GAT layer: output of gat [in_channels * heads, 12]
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTM
        # [batchsize * n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)
        x = torch.reshape(x, (batch_size, n_node, data.num_features))
        # for lstm: x should be (seq_length, batch_size, n_nodes)
        # sequence length = 24 (a full day), batch_size = 256, n_node = 33
        x = torch.movedim(x, 2, 0)
        # [24, 256, 33] -> [24, 256, 32] > lstm transformation
        x, _ = self.lstm1(x)
        # [24, 256, 32] -> [24, 256, 128] > lstm transformation
        x, _ = self.lstm2(x)

        # output contains h_t for each timestep, only the last one has all input's accounted for
        # [24, 256 ,128] -> [256, 128]
        x = torch.squeeze(x[-1, :, :])
        # [256, 128] -> [256, 33*9]
        x = self.linear(x)

        # now reshape into final output
        s = x.shape
        # [256, 228*9] -> [256, 228, 9]
        x = torch.reshape(x, (s[0], self.n_nodes, self.n_pred))
        # [256, 228, 9] -> [11400, 9]
        x = torch.reshape(x, (s[0] * self.n_nodes, self.n_pred))

        return x
