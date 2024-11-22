import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Import your dataset creation and model
from src.size_data_loader import create_dataset  
from models.TGAT_size import TGAT  

# Function to plot prediction vs. actual
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs. True Values")
    plt.show()

# Split data into training, validation, and test sets
def split_data(data_list):
    data_train, data_temp = train_test_split(data_list, test_size=0.3, random_state=42)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, random_state=42)
    return data_train, data_val, data_test

# Test model function with regression metrics
def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y = data.y.cpu().numpy()

            total_load_step = []
            sum_load = data.x.cpu().numpy()
            # print(sum_load[:,:,0])
            for i in range(24):
                load = sum_load[:,:,0][i].sum()
                total_load_step.append(load)

            # print(total_load_step)
            grid_profile = total_load_step * np.array(sum_load[:,:,1][1].min())
            lowest_x = np.argmin(grid_profile)


            # print(sum_load[:,:,1][1].min())
            plt.plot(total_load_step)
            plt.plot(grid_profile)

            output = model(data).squeeze()  # Output shape: (1,) -> scalar
            target = data.y.to(device).squeeze()  # Target shape: (1,) -> scalar

            plt.scatter(lowest_x, output.cpu(), color='red', label='Predicted Point')
            plt.scatter(lowest_x, target.cpu(), color='blue', label='Real Point')
            plt.annotate(f'({lowest_x}, {output.cpu():.2f})',
                         xy=(lowest_x, output.cpu()),
                         xytext=(lowest_x + 1, output.cpu() + 0.1),
                         arrowprops=dict(facecolor='black', arrowstyle='->'))
            plt.annotate(f'({lowest_x}, {target.cpu():.2f})',
                         xy=(lowest_x, target.cpu()),
                         xytext=(lowest_x + 1, target.cpu() + 0.1),
                         arrowprops=dict(facecolor='black', arrowstyle='->'))
            
            # Add legend and labels
            plt.legend()
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Graph with Lowest Point Highlighted')
            # # Show the plot
            plt.show()
            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Collect predictions and targets
            all_preds.append(output.cpu().item())
            all_targets.append(target.cpu().item())

    # Convert lists to numpy arrays for evaluation
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate regression metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot predictions vs. true values
    plot_predictions(all_targets, all_preds)

    return total_loss / len(loader.dataset)

# Define the criterion
criterion = torch.nn.MSELoss()

# Load dataset
data_list = create_dataset("data/sizing_dataset.h5")
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
n_classes = 4 

# Initialize the model
# model = TGAT(in_channels, hidden_channels).to(device)
model = TGAT(in_channels, hidden_channels, n_classes).to(device)

# Load the model state
model.load_state_dict(torch.load("checkpoints/best_model_size.pth"))

# Run the test function
test_loss = test_model(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f}")
