import torch

def train_model(model, data_loader, criterion, optimizer, device, epoch, checkpoint_path='checkpoints/model_epoch_{epoch}.pth', final_model_path='checkpoints/model_final.pth'):
    model.train()
    total_loss = 0
    all_attention_weights = []  # To collect attention weights from all batches
    for data, targets in data_loader:
        # Move data and targets to the correct device (GPU or CPU)
        # print(data, targets)
        data = data.to(device)
        targets = torch.cat([t.to(device) for t in targets])  # Concatenate node-level targets

        # Forward pass through the model
        # out, attention_weights = model(data)  # Model returns logits for each node
        out = model(data)  # Model returns logits for each node

        # Compute the loss using CrossEntropyLoss
        # `out` has shape [num_nodes, num_classes], and `targets` has shape [num_nodes]
        loss = criterion(out, targets)  # No need for torch.argmax on targets

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # all_attention_weights.append(attention_weights)

    average_loss = total_loss / len(data_loader)

    # Save the model and attention weights after every epoch
    torch.save(model.state_dict(), checkpoint_path.format(epoch=epoch+1))

    average_loss = total_loss / len(data_loader)
    # return average_loss, all_attention_weights
    return average_loss
