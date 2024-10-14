import torch
from src.data_loader import *
from models.GAT import GATNet  # Assuming your model is defined in train.py
import numpy as np
import time
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
from utils.constants import *
from torch_geometric.data import Data


network = nw.case33bw()

def load_data(net, time_step):
    # Reset loads to original before applying scaling factors
    ######### adding a "single season" based increase factor
    print(net.load[['p_mw', 'q_mvar']].values)
    net.load.loc[NODE_TYPE[1:] == 'residential', 'p_mw'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
    net.load.loc[NODE_TYPE[1:] == 'commerical', 'p_mw'] *= COMMERCIAL_LOAD_FACTOR[time_step]
    net.load.loc[NODE_TYPE[1:] == 'industrial', 'p_mw'] *= INDUSTRIAL_LOAD_FACTOR[time_step]

    # scaling reactive power (starting from [1:] to avoid slack bus)
    net.load.loc[NODE_TYPE[1:] == 'residential', 'q_mvar'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
    net.load.loc[NODE_TYPE[1:] == 'commerical', 'q_mvar'] *= COMMERCIAL_LOAD_FACTOR[time_step]
    net.load.loc[NODE_TYPE[1:] == 'industrial', 'q_mvar'] *= INDUSTRIAL_LOAD_FACTOR[time_step]
    print('*' *  100)
    print(f'Total residential nodes: {np.sum(NODE_TYPE == 'residential')},\nTotal commercial nodes: {np.sum(NODE_TYPE == 'commercial')},\nTotal industrial nodes: {np.sum(NODE_TYPE == 'industrial')}')

    vectorized_mapping = np.vectorize(CLASS_MAPPING.get)
    target_classes = vectorized_mapping(NODE_TYPE)

    # extract edge features from test case
    line_data = net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']].values
    edge_features = line_data[:, 2:5] 

    # extract node features (p and q)
    pp.runpp(net, verbose=True, numba=False)
    # node_features = net.load[['p_mw', 'q_mvar']].values
    node_features = net.res_bus[['p_mw', 'q_mvar']].values
    print(node_features)

    # create edge index  (from_bus and to_bus in first two columns)
    edge_index = np.vstack((line_data[:, 0], line_data[:, 1])).astype(int)

    # convert data to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=node_features, edge_index= edge_index, edge_attr=edge_features)

    return data, target_classes
        
# Load the trained model
def load_model(model_path):
    model = GATNet()  # Replace with your model class
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict classes of the provided data
def predict_classes(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        # predicted_classes = predictions.argmax(dim=1)
    return predictions

def main():
    # Load your trained model
    model_path = "checkpoints/best_model.pth"  # Update with your actual model path
    model = load_model(model_path)

    # Load your data (you might need to modify this based on your data format)
    for time_step in range(24):
        data, classes = load_data(network, time_step)  # Modify based on your data loader's function
        # Make predictions
        predicted_classes = predict_classes(model, data)

        # Display the predictions
        print("Predicted Classes for the input data:")
        print(predicted_classes[0])
        # print(np.argmax(predicted_classes[0]), classes)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
