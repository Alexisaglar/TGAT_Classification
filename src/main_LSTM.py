import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Updated to use torch.utils.data.DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your dataset creation function
from src.temp_data_loader import create_dataset  # Your data loading function

# Import the LSTM-only model
from models.LSTM import LSTMNodeClassifierWithAttention # Your LSTM-only model

# We'll define updated train and test functions below

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

def collate_fn(data_batch):
    # Custom collate function to prepare batch data
    # Since batch_size is 1, data_batch will be a list of length 1
    data = data_batch[0]
    X = data.x  # Shape: (seq_length, n_nodes, in_channels)
    y = data.y  # Shape: (n_nodes,)
    return X, y

def train_model(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for X, y in data_loader:
        X = X.permute(1, 0, 2).to(device)  # Shape: (n_nodes, seq_length, in_channels)
        y = y.to(device)  # Shape: (n_nodes,)

        optimizer.zero_grad()
        output = model(X)  # Output shape: (n_nodes, n_classes)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

def test_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.permute(1, 0, 2).to(device)  # Shape: (n_nodes, seq_length, in_channels)
            y = y.to(device)  # Shape: (n_nodes,)

            output = model(X)  # Output shape: (n_nodes, n_classes)
            loss = criterion(output, y)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This training is using {device}")

    # Load the dataset
    data_list = create_dataset("data/load_classification_100_networks.h5")
    data_train, data_val, data_test = split_data(data_list)
    batch_size = 1  # Since each data sample may have different sizes

    # Create DataLoader instances with custom collate_fn
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Determine the number of classes
    all_targets = torch.cat([data.y.view(-1) for data in data_train])
    n_classes = len(torch.unique(all_targets))
    print(f"Number of classes: {n_classes}")

    # Model parameters
    in_channels = 2       # Number of input features per node
    hidden_size = 64      # Hidden size for LSTM
    n_nodes = 33          # Number of nodes
    seq_length = 24       # Number of time steps
    
    # Initialize the model
    model = LSTMNodeClassifierWithAttention(in_channels, hidden_size, n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

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
    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
