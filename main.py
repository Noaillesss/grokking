from YParams import YParams
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from grokking.data import *
from grokking.model import *
import grokking.training as training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", nargs="?", type=str, default="transformer")   # mlp / lstm / transformer_k
    parser.add_argument("--training_fraction", nargs="?", type=float, default=0.5)
    parser.add_argument("--optimizer", nargs="?", type=str, default="adamw")            # adam / adamw / sgd
    parser.add_argument("--random_seed", nargs="?", type=int, default=42)
    parser.add_argument("--length", nargs="?", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = "config.yaml"
    architecture = args.architecture
    print("Using architecture", architecture)
    params = YParams(config_file, architecture, print_params=True)
    params.device = device

    if args.training_fraction is not None:
        params.training_fraction = args.training_fraction
        print("Overriding training fraction to", params.training_fraction)
    if args.optimizer is not None:
        params.optimizer = args.optimizer
        print("Overriding optimizer to", params.optimizer)
    if args.random_seed is not None:
        params.random_seed = args.random_seed
        print("Overriding random seed to", params.random_seed)
    if args.length is not None:
        params.length = args.length
        print("Overriding length to", params.length)
    
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    # Prepare the training and validation data
    train_loader, val_loader = get_data(
        params.length,
        params.prime,
        params.training_fraction,
        params.batch_size
        )
    
    if not hasattr(params, 'total_steps') or params.total_steps is None:
        params.total_steps = params.epochs * len(train_loader)
    if not hasattr(params, 'warmup_steps') or params.warmup_steps is None:
        params.warmup_steps = int(params.total_steps * 0.1)

    # Initialize the model
    if architecture == "transformer" or architecture == "transformer_k":
        model = Transformer(
            num_layers=params.num_layers,
            dim_model=params.dim_model,
            num_heads=params.num_heads,
            num_tokens=params.prime + 2,
            seq_len=2*params.length,
            dropout=params.dropout
            ).to(device)
    elif architecture == "mlp":
        model = MLP(
            num_layers=params.num_layers,
            dim_model=params.dim_model,
            num_heads=params.num_heads,
            num_tokens=params.prime + 2,
            seq_len=2*params.length,
            dropout=params.dropout
            ).to(device)
    elif architecture == "lstm":
        model = LSTM(
            num_layers=params.num_layers,
            dim_model=params.dim_model,
            num_tokens=params.prime + 2,
            hidden_dim=params.hidden_dim,
            dropout=params.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Train the model
    train_accuracy, train_loss, val_accuracy, val_loss = training.train(model, train_loader, val_loader, params)

    # Save the results
    save_fig_path = f'./figures/{architecture}'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)


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
    plt.savefig(save_fig_path + f'/{params.training_fraction * 100:.0f}_accuracy.png')

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
    plt.savefig(save_fig_path + f'/{params.training_fraction * 100:.0f}_loss.png')

    # plt.show()