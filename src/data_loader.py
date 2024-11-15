import numpy as np
import torch
from torch_geometric.data import Data
import h5py
import time

normalize = True

def normalize_min_max(tensor):
    # Min-Max Normalization
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def normalize_zscore(tensor):
    # Z-score Normalization
    return (tensor - tensor.mean()) / tensor.std()

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
                    time_step_targets = []
                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]

                        # Extract node features (p and q) for this time step
                        node_features = time_step_group['res_bus'][:, 1:3]  # Shape: (n_nodes, 2)
                        time_step_features.append(node_features)

                        targets = static_data['bus'][:, 3]  # Class labels for each node
                        time_step_targets.append(targets) 
                    
                    # Convert the list of node features over time to a tensor
                    time_step_features = np.stack(time_step_features, axis=0)  # Shape: (n_time_steps, n_nodes, 2)
                    time_step_targets = np.stack(time_step_targets, axis=0)  # Shape: (n_time_steps, n_nodes)

                    # Create edge index (remains constant for all time steps)
                    edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
                    edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

                    # Convert to torch tensors
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                    edge_features = torch.tensor(edge_features, dtype=torch.float)
                                        
                    # Generate sequences using a sliding window
                    for i in range(time_step_features.shape[0] - seq_length + 1):
                        node_feature_sequence = time_step_features[i:i+seq_length]  # Shape: (seq_length, n_nodes, num_features)
                        target_sequence = time_step_targets[i:i+seq_length]

                        # Convert to tensors
                        node_feature_sequence = torch.tensor(node_feature_sequence, dtype=torch.float)
                        target_sequence = torch.tensor(target_sequence, dtype=torch.long)

                        # Use the last target in the sequence for y
                        data = Data(x=node_feature_sequence, edge_index=edge_index, edge_attr=edge_features, y=target_sequence[-1])
                        data_list.append(data)
                        # print(f"Data object created with x shape: {data.x.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_features.shape}, y shape: {data.y.shape}")

        return data_list
