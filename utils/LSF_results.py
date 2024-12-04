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
from src.LSF_data_loader import create_dataset  
from models.TGAT_LSF import TGAT  

# Function to plot prediction vs. actual
def plot_predictions(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.7)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
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

            # Get model predictions
            output = model(data)  # Output shape: (n_classes,)
            target = data.y.to(device)  # Target shape: (n_classes,)

            # Ensure output and target have correct shapes
            output = output.view(-1)
            target = target.view(-1)

            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Collect predictions and targets
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # --- Calculate Total Load ---
            total_load_step = []
            sum_load = data.x.cpu().numpy()  # Adjust shape as necessary
            # Assuming sum_load shape is [num_timesteps, num_nodes, num_features]
            num_timesteps = sum_load.shape[0]
            for i in range(num_timesteps):
                # Sum over nodes for each timestep
                load = sum_load[i, :, 0].sum()  # Assuming the 0th feature is load
                total_load_step.append(load)

            # --- Calculate Grid Profile ---
            # Assuming the 1st feature (index 1) is relevant for grid profile calculation
            min_value = sum_load[:, :, 1].min()
            grid_profile = np.array(total_load_step) * min_value

            # --- Find the Lowest Point in Grid Profile ---
            lowest_x = np.argmin(grid_profile)

            # --- Plot Total Load and Grid Profile ---
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_timesteps), total_load_step, label='Total Load')
            plt.plot(range(num_timesteps), grid_profile, label='Grid Profile')
            plt.scatter(lowest_x, grid_profile[lowest_x], color='red', label='Lowest Point')
            plt.annotate(f'({lowest_x}, {grid_profile[lowest_x]:.2f})',
                         xy=(lowest_x, grid_profile[lowest_x]),
                         xytext=(lowest_x + 1, grid_profile[lowest_x] + 0.1),
                         arrowprops=dict(facecolor='black', arrowstyle='->'))
            plt.xlabel('Time Step')
            plt.ylabel('Load')
            plt.title('Total Load and Grid Profile Over Time')
            plt.legend()
            plt.show()

            # --- Plot Predicted vs. True Values for This Sample ---
            plt.figure(figsize=(10, 6))
            plt.plot(range(33), target.cpu().numpy(), label='True')
            plt.plot(range(33), output.cpu().numpy(), label='Predicted')
            plt.xlabel('Output Index')
            plt.ylabel('Value')
            plt.title('Predicted vs True Values for Sample')
            plt.legend()
            plt.show()

    # Convert lists to numpy arrays for evaluation
    all_preds = np.vstack(all_preds)  # Shape: (num_samples, n_classes)
    all_targets = np.vstack(all_targets)

    # Flatten for metric calculations
    all_preds_flat = all_preds.flatten()
    all_targets_flat = all_targets.flatten()

    # Calculate regression metrics
    mse = mean_squared_error(all_targets_flat, all_preds_flat)
    mae = mean_absolute_error(all_targets_flat, all_preds_flat)
    r2 = r2_score(all_targets_flat, all_preds_flat)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot predictions vs. true values across all samples
    plot_predictions(all_targets_flat, all_preds_flat)

    return total_loss / len(loader)

# Define the criterion
criterion = torch.nn.MSELoss()

# Load dataset
data_list = create_dataset("data/full_100_network_dataset.h5")
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
n_classes = 16  # Changed from 4 to 16 to match the new output size

# Initialize the model
model = TGAT(in_channels, hidden_channels, 4).to(device)

# Load the model state
model.load_state_dict(torch.load("checkpoints/best_model_size.pth"))

# Run the test function
test_loss = test_model(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f}")
