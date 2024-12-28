from YParams import YParams
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from grokking.data import *
from grokking.model import *
import grokking.training as training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="?", type=str, default="config.yaml")
    parser.add_argument("--architecture", nargs="?", type=str, default="transformer")
    parser.add_argument("--optimizer", nargs="?", type=str, default="adamw")
    parser.add_argument("--random_seed", nargs="?", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = args.config
    architecture = args.architecture
    print("Using architecture", architecture)
    params = YParams(config_file, architecture, print_params=True)
    params.device = device

    if args.random_seed is not None:
        params.random_seed = args.random_seed
        print("Overriding random seed to", params.random_seed)
    
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    train_loader, val_loader = get_data(
        params.operation,
        params.prime,
        params.training_fraction,
        params.batch_size
        )
    
    if architecture == "transformer":
        model = Transformer(
            num_layers=params.num_layers,
            dim_model=params.dim_model,
            num_heads=params.num_heads,
            num_tokens=params.prime + 2,
            seq_len=5
            ).to(device)
    elif architecture == "mlp":
        NotImplementedError("MLP architecture not implemented")
    elif architecture == "lstm":
        NotImplementedError("LSTM architecture not implemented")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
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
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters=9
    )

    train_accuracy, train_loss, val_accuracy, val_loss = training.train(model, train_loader, val_loader, optimizer, scheduler, params)

    # Plot the training and validation accuracy
    steps = np.arange(1, len(train_accuracy) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_accuracy, 'r-', label='train', linewidth=2)
    plt.plot(steps, val_accuracy, 'g-', label='val', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy')
    plt.title(f'Modular Sum (training on {params.training_fraction * 100:.0f}% of data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./figures/{architecture}_{params.training_fraction * 100:.0f}_accuracy.png')

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, 'r-', label='train', linewidth=2)
    plt.plot(steps, val_loss, 'g-', label='val', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss')
    plt.title(f'Modular Sum (training on {params.training_fraction * 100:.0f}% of data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./figures/{architecture}_{params.training_fraction * 100:.0f}_loss.png')

    # plt.show()