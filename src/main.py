from src.test import test_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.data_loader import create_dataset
from models.GAT import GATNet
from src.train import train_model


def split_data(data_list, target_list):
    data_train, data_temp, target_train, target_temp = train_test_split(
        data_list, target_list, test_size=0.3, random_state=42
    )
    data_val, data_test, target_val, target_test = train_test_split(
        data_temp, target_temp, test_size=0.5, random_state=42
    )
    return data_train, data_val, data_test, target_train, target_val, target_test


def plot_performance(train_loss, val_loss):
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
    data_list, target_list = create_dataset("data/multimodal_example.h5")
    data_train, data_val, data_test, target_train, target_val, target_test = split_data(
        data_list, target_list
    )

    batch_size = 256 

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

    model = GATNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    patience = 15
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(10000):
        train_loss, train_attention_weights = train_model(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_attention_weights = test_model(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()

