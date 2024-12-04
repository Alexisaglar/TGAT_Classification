import numpy as np
import torch
from torch_geometric.data import Data
import h5py
import time
import matplotlib.pyplot as plt
import pandas as pd

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
            # 'pv_potential': net_group['network_config/pv_potential'][:]
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
            # season_load_sums = []
            # season_pv_potential = []
            season_sensitivity = []

            for season_key in season_keys:
                season = int(season_key.split('_')[-1])

                if season_key in net_group:
                    season_group = net_group[season_key]

                    time_step_features = []
                    time_step_load_sums = []
                    time_step_sensitivity = []
                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]

                        time_step = int(time_step_key.split('_')[-1])


                        # Extract node features 
                        loads = time_step_group['res_bus'][:, 2:3]  
                        # remove negative load from slack bus
                        loads = np.where(loads > 0, loads, 0) 
                        # grid_factor = static_data['bus'][:, 4:5]  
                        sensitivity_t = time_step_group['res_sensitivity'][:]
                        # print(sensitivity_t)
                        # # print(pd.DataFrame(static_data['pv_potential']).iloc[:,::4])
                        # print(pv_potential_t)
                        # # repeat same values for all buses
                        delta_p = np.tile(0.1, (33,1))
                        node_features = np.hstack((loads, delta_p))


                        # Calculate load sum (p) scaled by grid factor
                        # load_sum = np.sum(loads)  # Single scalar value
                        # grid_required = np.sum(loads * grid_factor)  # Single scalar value
                        
                        time_step_features.append(node_features)
                        # time_step_load_sums.append(load_sum)
                        time_step_sensitivity.append(sensitivity_t)



                # remain_load = np.array(time_step_load_sums) - np.array(time_step_grid_req)
                # max_pv_potentials = np.max(static_data['pv_potential'][:, season::4].T, axis=1)
                # indices_max_pv_potentials = np.argmax(static_data['pv_potential'][:, season::4].T, axis=1)
                
                # print(np.shape(np.array(time_step_sensitivity)))
                bus_mean_sensitivity = np.mean(np.array(time_step_sensitivity), axis=0)
                # print(np.mean(np.array(time_step_sensitivity), axis=0))
                #
                # for i in range(time_step_sensitivity.shape[0]):
                #     # load_pv_potential = remain_load[indices_max_pv_potentials[i]] / max_pv_potentials[i]
                #     # season_pv_potential.append(load_pv_potential)
                #
                #     print(i)

                # print(season_pv_potential)
                season_sensitivity.append(bus_mean_sensitivity)
                # season_load_sums.append(time_step_load_sums)
                season_features.append(time_step_features) 

            target = []
            print(np.mean(np.array(season_sensitivity), axis = 0))
            target = np.mean(np.array(season_sensitivity), axis = 0)
            # target = np.divide(season_pv_potential , 100)

            print(target)
            season_features = np.stack(season_features, axis=0)
            seasons = 4
            num_nodes= 33
            node_features = 2
            # print(np.shape(season_features))
            season_features = season_features.reshape((seq_length*seasons), num_nodes, node_features)
            print(season_features)


            # Create edge index (remains constant for all time steps)
            edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
            edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

            # Convert to torch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
                                
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
