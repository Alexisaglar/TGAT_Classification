import torch

def test_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    all_attention_weights = []  # To collect attention weights from all batches
   
    with torch.no_grad():  # No gradients are needed
        for data, targets in val_loader:
            data = data.to(device)
            # targets = targets.view(-1,1)
            targets = torch.cat([t.to(device).long() for t in targets])  # Ensure targets are LongTensor

            out, attention_weights = model(data)
            loss = criterion(out, targets)
            total_val_loss += loss.item()

            all_attention_weights.append(attention_weights)

    average_val_loss = total_val_loss / len(val_loader)
    return average_val_loss, all_attention_weights

