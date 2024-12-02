import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your dataset creation and model
from src.size_data_loader import create_dataset  # Your data loading function
from models.TGAT_size import TGAT  # Your TemporalGAT model
from src.train import train_model  # Your training and evaluation functions
from src.test import test_model

def split_data(data_list):
    # Split the dataset into training, validation, and test sets
    data_train, data_temp = train_test_split(data_list, test_size=0.3, random_state=42)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, random_state=42)

    return data_train, data_val, data_test

def plot_performance(train_loss, val_loss):
    # Plot the training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This training is using {device}")

    # Load the dataset
    # data_list = create_dataset("data/size_dataset_100.h5")
    data_list = create_dataset("data/power_potential_networks.h5")
    data_train, data_val, data_test = split_data(data_list)
    batch_size = 1  # Adjust according to your available memory

    # Create DataLoader instances
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    # Determine the number of classes
    all_targets = torch.cat([data.y.view(-1) for data in data_train])
    n_classes = len(torch.unique(all_targets))
    # print(f"Number of classes: {n_classes}")

    # Model parameters
    in_channels = 6  # Number of input features per node
    hidden_channels = 128  # Hidden size for GAT layers
    n_nodes = 33  # Number of nodes in the graph
    seq_length = 96  # Number of time steps
    n_classes = 4
    
    # Initialize the model
    model = TGAT(in_channels, hidden_channels, n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    patience = 15 
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1000):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = test_model(model, val_loader, criterion, device)
        # Store loss for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping mechanism
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model_size.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        print(f"Epoch {epoch + 2}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot performance
    plot_performance(train_losses, val_losses)

    # Load the best model for testing
    model.load_state_dict(torch.load("checkpoints/best_model_size.pth"))
    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
if __name__ == "__main__":
    main()
