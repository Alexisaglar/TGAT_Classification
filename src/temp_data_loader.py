import numpy as np
import torch
from torch_geometric.data import Data
import h5py
import time

def load_network_data(file, network_key):
    with h5py.File(file, 'r') as f:
        net_group = f[network_key]
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:]
        }
    return static_data

def create_dataset(file, seq_length=24):
    data_list = []
    target_list = []

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_0', 'season_1', 'season_2', 'season_3']
            
            for season_key in season_keys:
                if season_key in net_group:
                    season_group = net_group[season_key]

                    # Collect node features over time steps
                    time_step_features = []
                    for time_step_key in season_group.keys():
                        time_step_group = season_group[time_step_key]

                        # Extract node features (p and q) for this time step
                        node_features = time_step_group['res_bus'][:, 1:3]  # Shape: (n_nodes, 2)
                        time_step_features.append(node_features)
                    
                    # Convert the list of node features over time to a tensor
                    # time_step_features = np.stack(time_step_features, axis=0)  # Shape: (n_time_steps, n_nodes, 2)

                    # If there are fewer time steps than seq_length, skip this sequence
                    # if time_step_features.shape[0] < seq_length:
                    #     continue
                    
                    # Collect node feature sequences of length `seq_length`
                    # for i in range(time_step_features.shape[0]):
                    # print(time_step_features.shape)
                    node_feature_sequence = time_step_features  # Shape: (seq_length, n_nodes, 2)
                    # print(f'node_feature_sequence: seq_length: {seq_length}, i:{i}')
                    
                    # Create edge index (remains constant for all time steps)
                    edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
                    edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

                    edge_index = edge_index - edge_index.min()  # Reindex to [0, 32]
                    print(f'this is edge_index array: {edge_index}')

                    # Extract target values (voltage magnitude and angle) for classification
                    target_bus = static_data['bus'][:, 3]  # Class labels for each node
                    
                    # Convert to torch tensors
                    node_feature_sequence = torch.tensor(node_feature_sequence, dtype=torch.float)
                    edge_features = torch.tensor(edge_features, dtype=torch.float)
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                    print(f'this is edge_index tensor: {edge_index}')
                    targets = torch.tensor(target_bus, dtype=torch.long)
                    
                    # Create Data object (node features are now sequences over time)
                    data = Data(x=node_feature_sequence, edge_index=edge_index, edge_attr=edge_features)
                    print(edge_index)
                    data_list.append(data)
                    target_list.append(targets)

                    # print(node_features.shape, node_feature_sequence.shape, edge_features.shape, )
    return data_list, target_list
