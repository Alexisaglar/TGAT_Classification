import torch
import pandas as pd

from models.trainer import load_from_checkpoint, model_train, model_test
from torch_geometric.loader import DataLoader
from data_loader.dataloader import TrafficDataset, get_splits, distance_to_weight
from utils.constants import *

def main():
    """
    Main function to train and test a model.
    """

    # Number of possible windows in a day
    CONFIG['N_SLOT']= CONFIG['N_DAY_SLOT'] - (CONFIG['N_PRED']+ CONFIG['N_HIST']) + 1

    # Load the weight matrix
    distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
    W = distance_to_weight(distances, gat_version=CONFIG['USE_GAT_WEIGHTS'])
    # Load the dataset
    dataset = TrafficDataset(CONFIG, W)

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    train, val, test = get_splits(dataset, CONFIG['N_SLOT'], (34, 5, 5))
    train_dataloader = DataLoader(train, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    # Get gpu if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Configure and train model
    CONFIG['N_NODE'] = dataset.n_node
    model = model_train(train_dataloader, val_dataloader, CONFIG, device)
    # Or, load from a saved checkpoint
    # model = load_from_checkpoint('./runs/model_final_60epochs.pt', CONFIG)
    # Test Model
    model_test(model, test_dataloader, device, CONFIG)


if __name__ == "__main__":
    main()
