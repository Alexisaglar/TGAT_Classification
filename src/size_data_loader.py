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
            'bus': net_group['network_config/bus'][:],
            'pv_potential': net_group['network_config/pv_potential'][:]
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
            season_features = []
            season_load_sums = []
            season_pv_potential = []

            for season_key in season_keys:
                season = int(season_key.split('_')[-1])
                start_col = season * 4
                end_col = (season + 1) * 4

                if season_key in net_group:
                    season_group = net_group[season_key]

                    time_step_features = []
                    time_step_load_sums = []
                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]

                        time_step = int(time_step_key.split('_')[-1])


                        # Extract node features 
                        loads = time_step_group['res_bus'][:, 2:3]  
                        # remove negative load from slack bus
                        loads = np.where(loads > 0, loads, 0) 
                        grid_factor = static_data['bus'][:, 4:5]  
                        pv_potential_t = static_data['pv_potential'][time_step, start_col:end_col]
                        # repeat same values for all buses
                        pv_potential_buses = np.tile(pv_potential_t, (33,1))
                        node_features = np.hstack((loads, grid_factor, pv_potential_buses))


                        # Calculate load sum (p) scaled by grid factor
                        load_sum = np.sum(loads * grid_factor)  # Single scalar value
                        
                        time_step_features.append(node_features)
                        time_step_load_sums.append(load_sum)

                load_pv_potential = np.array(time_step_load_sums)[np.newaxis,:] * static_data['pv_potential'][:, start_col:end_col].T
                season_pv_potential.append(load_pv_potential)
                # seasonal_pv_potential = np.sum(seasonal_pv_potential,axis=1)
                # print(season_pv_potential)
                season_load_sums.append(time_step_load_sums)
                season_features.append(time_step_features) 
            # time.sleep(5)
            print(np.shape(season_pv_potential))
            target = []
            season_pv_potential = np.array(season_pv_potential)
            for season in range(season_pv_potential.shape[0]):
                for technology in range(season_pv_potential.shape[1]):
                    max_value = np.max(season_pv_potential[season, technology, :])
                    print(max_value)
                    target.append(max_value)

            print(target)
            season_features = np.stack(season_features, axis=0)
            seasons = 4
            num_nodes= 33
            node_features = 6
            season_features = season_features.reshape((seq_length*seasons), num_nodes, node_features)

            # Convert the list of node features over time to a tensor
            # time_step_features = np.stack(time_step_features, axis=0)  # Shape: (n_time_steps, n_nodes, 3)
            # print(time_step_features)
            # time.sleep(5)
            # time_step_load_sums = np.array(time_step_load_sums)  # Shape: (n_time_steps,)

            # Create edge index (remains constant for all time steps)
            edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
            edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

            # Convert to torch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
                                
            # Generate sequences using a sliding window
            # for i in range(season_features.shape[0] - seq_length + 1):
            #     node_feature_sequence = time_step_features[i:i+seq_length]  # Shape: (seq_length, n_nodes, 3)
            #     load_sum_sequence = time_step_load_sums[i:i+seq_length]  # Shape: (seq_length,)
            #
            #     # Calculate the target: minimum load sum over the sequence
            #     target = np.min(load_sum_sequence)  # Single scalar value
            #     # print(target)
            #    
            #     # Convert to tensors
            node_feature_sequence = torch.tensor(season_features, dtype=torch.float)
            target = torch.tensor(target, dtype=torch.float)  # Target as scalar float

            # Create the Data object
            data = Data(
                x=node_feature_sequence,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=target
            )
            
            data_list.append(data)

        print(data_list)
        return data_list
