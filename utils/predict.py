import time 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Import your dataset creation and model
from src.temp_data_loader import create_dataset  # Your data loading function
from models.TGAT import TGAT  # Your TemporalGAT model
from src.train import train_model  # Your training and evaluation functions
from src.test import test_model

def plot_real_vs_pred(real, pred, n_nodes):
    # Create a plot for real vs predicted classes
    plt.figure(figsize=(10, 5))
    
    # Plot real classes
    plt.plot(range(n_nodes), real, color='blue', label='Real')
    plt.plot(range(n_nodes), pred, color='red', label='Predicted')
    plt.title('Real Classes')
    plt.xlabel('Node')
    plt.ylabel('Class')
    plt.xticks(range(n_nodes))
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.show()


def split_data(data_list):
    # Split the dataset into training, validation, and test sets
    data_train, data_temp = train_test_split(data_list, test_size=0.3, random_state=42)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, random_state=42)

    return data_train, data_val, data_test


def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            target = data.y.to(device)

            # # Reshape for loss computation
            # output = output.view(-1, output.size(-1))
            # target = target.view(-1)
            #
            print(target.shape)
            print(output)
            clase = 10 
            classes = []
            for _ in range(33):
                pass
            for items in output:
                print(np.where(items == items.max())[0])
                clase = np.where(items == items.max())[0][0]
                classes.append(clase)


            plot_real_vs_pred(target, classes, 33)
            loss = criterion(output, target)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

criterion = torch.nn.CrossEntropyLoss()



data_list = create_dataset("data/load_classification_100_networks.h5")
data_train, data_val, data_test = split_data(data_list)
batch_size = 1  # Adjust according to your available memory

# Create DataLoader instances
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This training is using {device}")
# Model parameters
in_channels = 2  # Number of input features per node
hidden_channels = 128  # Hidden size for GAT layers
n_nodes = 33  # Number of nodes in the graph
seq_length = 24  # Number of time steps
n_classes = 4  # Number of time steps

# Initialize the model
model = TGAT(in_channels, hidden_channels, n_classes).to(device)

model.load_state_dict(torch.load("checkpoints/best_model.pth"))
test_loss = test_model(model, test_loader, criterion, device)
print(f"test loss: {test_loss:.4f}")

