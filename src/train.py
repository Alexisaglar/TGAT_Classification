import torch

def train_model(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)  # Shape: (batch_size, n_nodes, n_classes)
        target = data.y.to(device)  # Shape: (batch_size, n_nodes)

        # # Reshape for loss computation
        # output = output.view(-1, output.size(-1))  # Shape: (batch_size * n_nodes, n_classes)
        # target = target.view(-1)  # Shape: (batch_size * n_nodes)

        # print(target)
        # print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        # torch.save(model.state_dict(), checkpoint_path.format(epoch=epoch+1))

    return total_loss / len(loader.dataset)

