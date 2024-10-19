import torch

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
            loss = criterion(output, target)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)
