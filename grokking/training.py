import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, params):
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n], "weight_decay": params.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "bias" in n or "norm" in n], "weight_decay": 0.0},
    ]
    # set up optimizer
    if params.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=params.weight_decay
            )
    elif params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.0001
            )
    elif params.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=params.learning_rate,
            momentum=0
            )
    elif params.optimizer == "sgd_momentum":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=params.learning_rate,
            momentum=params.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {params.optimizer}")
    
    def lr_lambda(current_step: int):
        if current_step < params.warmup_steps:
            return current_step / params.warmup_steps
        else:
            progress = (current_step - params.warmup_steps) / max(1, params.total_steps - params.warmup_steps)
            progress_tensor = torch.tensor(progress, dtype=torch.float32)
            return 0.5 * (1 + torch.cos(progress_tensor * 3.141592653589793))
    
    # set up scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) if params.optimizer == "sgd" or params.optimizer == "sgd_momentum" else torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, total_iters=10)

    device = params.device
    epochs = params.epochs
    log_test_interval = params.log_test_interval if hasattr(params, 'log_test_interval') else epochs + 1

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    current_step = 0

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
            if params.optimizer == "sgd" or params.optimizer == "sgd_momentum":
                scheduler.step()
                current_step += 1

            epoch_loss += loss.item()
        
        assert train_count == len(train_loader.dataset)
        avg_epoch_loss = epoch_loss / train_count
        train_accuracy.append(train_acc / train_count)
        train_loss.append(avg_epoch_loss)
        
        val_acc, test_loss = evaluate(model, val_loader, device)
        val_accuracy.append(val_acc)
        val_loss.append(test_loss)
        
        if not (params.optimizer == "sgd" or params.optimizer == "sgd_momentum"):
            scheduler.step()

        if (epoch + 1) % log_test_interval == 0:
            print(f"Epoch {epoch+1}: train loss: {avg_epoch_loss:.7f}, test loss: {test_loss:.7f}, learning rate: {scheduler.get_last_lr()[0]:.7f}")

        # Early stopping (for task 1 & 2)
        if val_acc > 0.99:
            print(f"Early stopping at epoch {epoch + 1}.")
            break
    
    # Print the best validation accuracy (for task 3)
    print(f"The best validation accuracy is {max(val_accuracy)} during {min(epoch+1,epochs)} epochs.")

    return train_accuracy, train_loss, val_accuracy, val_loss


def evaluate(model, val_loader, device):
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
