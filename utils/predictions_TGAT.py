import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from models.TGAT import TGAT
from src.temp_data_loader import create_dataset  # Your data loading function


def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            target = data.y.to(device)

            # Reshape for loss computation
            output = output.view(-1, output.size(-1))
            target = target.view(-1)

            loss = criterion(output, target)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def plot_real_vs_pred(real, pred, n_nodes):
    # Create a plot for real vs predicted classes
    plt.figure(figsize=(10, 5))
    
    # Plot real classes
    plt.subplot(1, 2, 1)
    plt.bar(range(n_nodes), real, color='blue', label='Real')
    plt.title('Real Classes')
    plt.xlabel('Node')
    plt.ylabel('Class')
    plt.xticks(range(n_nodes))
    plt.grid(True)
    
    # Plot predicted classes
    plt.subplot(1, 2, 2)
    plt.bar(range(n_nodes), pred, color='orange', label='Predicted')
    plt.title('Predicted Classes')
    plt.xlabel('Node')
    plt.ylabel('Class')
    plt.xticks(range(n_nodes))
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This script is using {device}")
    
    # Load the dataset
    data_list = create_dataset("data/load_classification_100_networks.h5")
    batch_size = 1  # Only one graph at a time

    # Create DataLoader instance
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    # Get the first graph from the DataLoader
    data = next(iter(loader)).to(device)

    # Determine the number of classes
    n_classes = len(torch.unique(data.y))
    n_nodes = data.num_nodes

    # Model parameters
    in_channels = 33  # Number of input features per node
    hidden_channels = 64  # Hidden size for GAT layers

    # Initialize the model
    model = TGAT(in_channels, hidden_channels, n_nodes, n_classes).to(device)
    # load the best model for testing
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_loss = test_model(model, data, criterion, device)
    print(f"test loss: {test_loss:.4f}")

    # Plot real vs predicted classes
    plot_real_vs_pred(real, pred, n_nodes)

if __name__ == "__main__":
    main()
