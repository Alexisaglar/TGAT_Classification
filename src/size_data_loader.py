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

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_0', 'season_1', 'season_2', 'season_3']
            
            for season_key in season_keys:
                if season_key in net_group:
                    season_group = net_group[season_key]

                    # Collect node features and load sums over time steps
                    time_step_features = []
                    time_step_load_sums = []
                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]

                        # Extract node features 
                        loads = time_step_group['res_bus'][:, 2:3]  
                        # print(loads)
                        # remove negative load from slack bus
                        loads = np.where(loads > 0, loads, 0) 
                        grid_factor = static_data['bus'][:, 4:5]  
                        node_features = np.hstack((loads, grid_factor))
                        # print()


                        # Calculate load sum (p and q combined) scaled by grid factor
                        load_sum = np.sum(loads * grid_factor)  # Single scalar value
                        # print(load_sum)
                        # time.sleep(1)
                        
                        time_step_features.append(node_features)
                        time_step_load_sums.append(load_sum)

                    # Convert the list of node features over time to a tensor
                    time_step_features = np.stack(time_step_features, axis=0)  # Shape: (n_time_steps, n_nodes, 3)
                    time_step_load_sums = np.array(time_step_load_sums)  # Shape: (n_time_steps,)

                    # Create edge index (remains constant for all time steps)
                    edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
                    edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

                    # Convert to torch tensors
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                    edge_features = torch.tensor(edge_features, dtype=torch.float)
                                        
                    # Generate sequences using a sliding window
                    for i in range(time_step_features.shape[0] - seq_length + 1):
                        node_feature_sequence = time_step_features[i:i+seq_length]  # Shape: (seq_length, n_nodes, 3)
                        load_sum_sequence = time_step_load_sums[i:i+seq_length]  # Shape: (seq_length,)

                        # Calculate the target: minimum load sum over the sequence
                        target = np.min(load_sum_sequence)  # Single scalar value
                        # print(target)
                        
                        # Convert to tensors
                        node_feature_sequence = torch.tensor(node_feature_sequence, dtype=torch.float)
                        target = torch.tensor(target, dtype=torch.float)  # Target as scalar float

                        # Create the Data object
                        data = Data(
                            x=node_feature_sequence,
                            edge_index=edge_index,
                            edge_attr=edge_features,
                            y=target
                        )
                        data_list.append(data)

        return data_list
