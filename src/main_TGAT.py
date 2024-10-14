import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your dataset creation and model
from src.temp_data_loader import create_dataset  # Your data loading function
from models.TGAT_example import TemporalGATNet  # Your TemporalGAT model
from src.train import train_model  # Your training and evaluation functions
from src.test import test_model

def split_data(data_list, target_list):
    # Split the dataset into training, validation, and test sets
    data_train, data_temp, target_train, target_temp = train_test_split(
        data_list, target_list, test_size=0.3, random_state=42
    )
    data_val, data_test, target_val, target_test = train_test_split(
        data_temp, target_temp, test_size=0.5, random_state=42
    )
    return data_train, data_val, data_test, target_train, target_val, target_test

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

    # Load the dataset (ensure it contains temporal features)
    data_list, target_list = create_dataset("data/multimodal_example.h5")
    data_train, data_val, data_test, target_train, target_val, target_test = split_data(
        data_list, target_list
    )

    batch_size = 256  # Adjust according to your available memory

    # Create DataLoader instances for batching
    train_loader = DataLoader(
        list(zip(data_train, target_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        list(zip(data_val, target_val)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
    )
    test_loader = DataLoader(
        list(zip(data_test, target_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
    )

    # Model parameters
    in_channels = 2  # Number of input features per node (adjust as needed)
    hidden_channels = 16  # Hidden size for GAT layers
    n_nodes = 33  # Number of nodes in the graph (adjust to your dataset)
    seq_length = 24  # Number of time steps (adjust to your temporal data)
    n_classes = 4  # Number of output classes (adjust based on your task)
    
    # Initialize the model
    model = TemporalGATNet(in_channels=in_channels, hidden_channels=hidden_channels, 
                           n_nodes=n_nodes, seq_length=seq_length, n_classes=n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    patience = 15
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(10000):
        train_loss, train_attention_weights = train_model(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_attention_weights = test_model(model, val_loader, criterion, device)

        # Store loss for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping mechanism
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot performance
    plot_performance(train_losses, val_losses)

    # Load the best model for testing
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_loss, test_attention_weights = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
