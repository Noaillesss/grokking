import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, params):
    # set up optimizer
    if params.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=params.weight_decay
            )
    elif params.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay
            )
    else:
        raise ValueError(f"Unknown optimizer: {params.optimizer}")
    
    # set up scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, total_iters=10)

    device = params.device
    epochs = params.epochs
    log_test_interval = params.log_test_interval if hasattr(params, 'log_test_interval') else epochs + 1

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    for epoch in tqdm(range(epochs)):
        # Set model to training mode
        model.train()
        epoch_loss = 0
        train_count = 0
        train_acc = 0
        criterion = torch.nn.CrossEntropyLoss()

        # Loop over each batch from the training set
        for batch in train_loader:

            # Copy data to device if needed
            batch = tuple(t.to(device) for t in batch)

            # Unpack the batch from the loader
            inputs, labels = batch
            
            # Zero gradient buffers
            optimizer.zero_grad()
        
            # Forward pass
            output = model(inputs)

            train_count += len(labels)
            loss = criterion(output, labels)
            train_acc += (torch.argmax(output, dim=1) == labels).sum().item()
        
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()
        
        assert train_count == len(train_loader.dataset)
        avg_epoch_loss = epoch_loss / train_count
        train_accuracy.append(train_acc / train_count)
        train_loss.append(avg_epoch_loss)
        
        val_acc, test_loss = evaluate(model, val_loader, device, params._config_name)
        val_accuracy.append(val_acc)
        val_loss.append(test_loss)
        
        scheduler.step()

        if (epoch + 1) % log_test_interval == 0:
            print(f"Epoch {epoch+1}: train loss: {avg_epoch_loss:.7f}, test loss: {test_loss:.7f}, learning rate: {scheduler.get_last_lr()[0]:.7f}")

        # Early stopping (for task 1 & 2)
        if val_acc > 0.99:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_accuracy, train_loss, val_accuracy, val_loss


def evaluate(model, val_loader, device, architecture):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0

    # Loop over each batch from the validation set
    for batch in val_loader:
        
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch
        
        # Forward pass
        with torch.no_grad():
            output = model(inputs)
            correct += (torch.argmax(output, dim=1) == labels).sum().item()
            loss += criterion(output, labels) * len(labels)
    
    acc = correct / len(val_loader.dataset)
    loss = loss.item() / len(val_loader.dataset)

    return acc, loss
