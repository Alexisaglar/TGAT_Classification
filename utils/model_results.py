import time 
import time 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Import your dataset creation and model
from src.temp_data_loader import create_dataset  # Your data loading function
from models.TGAT import TGAT  # Your TemporalGAT model
from src.train import train_model  # Your training and evaluation functions
from src.test import test_model

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, n_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot classification report
def plot_classification_report(y_true, y_pred, n_classes=4):
    report = classification_report(y_true, y_pred, labels=range(n_classes), output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Plot the classification report
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()

# Split data into training, validation, and test sets
def split_data(data_list):
    data_train, data_temp = train_test_split(data_list, test_size=0.3, random_state=42)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, random_state=42)
    return data_train, data_val, data_test

# Test model function with confusion matrix and classification report
def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            target = data.y.to(device)

            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss.item() * data.num_graphs

            # Get predicted classes
            pred_classes = output.argmax(dim=1).cpu().numpy()
            true_classes = target.cpu().numpy()

            all_preds.extend(pred_classes)
            all_targets.extend(true_classes)

    # Convert lists to arrays for confusion matrix and report
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Plot confusion matrix and classification report
    plot_confusion_matrix(all_targets, all_preds)
    plot_classification_report(all_targets, all_preds)

    return total_loss / len(loader.dataset)

# Define the criterion
criterion = torch.nn.CrossEntropyLoss()

# Load dataset
data_list = create_dataset("data/load_classification_100_networks.h5")
data_train, data_val, data_test = split_data(data_list)
batch_size = 1  # Adjust according to your available memory

# Create DataLoader instances
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This training is using {device}")

# Model parameters
in_channels = 2  # Number of input features per node
hidden_channels = 128  # Hidden size for GAT layers
n_nodes = 33  # Number of nodes in the graph
seq_length = 24  # Number of time steps
n_classes = 4  # Number of classes

# Initialize the model
model = TGAT(in_channels, hidden_channels, n_classes).to(device)

# Load the model state
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Run the test function
test_loss = test_model(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f}")

