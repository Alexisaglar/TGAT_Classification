import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        # Define each GATv2Conv layer separately
        self.conv_layer1 = GATv2Conv(in_channels=2, out_channels=16, heads=8, concat=True, edge_dim=3)
        self.conv_layer2 = GATv2Conv(in_channels=16*8, out_channels=64, heads=8, concat=True, edge_dim=3)
        self.conv_layer3 = GATv2Conv(in_channels=64*8, out_channels=32, heads=4, concat=True, edge_dim=3)
        self.conv_layer4 = GATv2Conv(in_channels=32*4, out_channels=4, heads=1, concat=False, edge_dim=3)


    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Process each layer sequentially
        X = self.conv_layer1(X, edge_index, edge_attr)
        X = F.elu(X)

        X = self.conv_layer2(X, edge_index, edge_attr)
        X = F.elu(X)
        
        X = self.conv_layer3(X, edge_index, edge_attr)
        X = F.elu(X)

        X, attention_weights = self.conv_layer4(X, edge_index, edge_attr, return_attention_weights=True)

        # X = F.log_softmax(X, dim=-1)
        # print(f"Output logits shape: {X.shape}")  # Add this to debug
        
        return X, attention_weights

