from math import ceil
import torch

def sum_mod_p_data(K: int, p: int, eq_token: int, op_token: int):
    """
    x_1 + x_2 + ... + x_K (mod p) for 0 <= x_i < p
    """
    x = torch.arange(0, p)
    x = torch.cartesian_prod(*[x for _ in range(K)])

    eq = torch.ones(p**K) * eq_token
    op = torch.ones(p**K) * op_token

    labels = x.sum(dim=1) % p

    inputs = torch.empty(p**K, 2*K, dtype=x.dtype)
    for i in range(K):
        inputs[:, 2*i] = x[:, i]
        inputs[:, 2*i+1] = op
    inputs[:, -1] = eq

    return inputs, labels

def get_data(K: int, prime: int, training_fraction: float, batch_size: int):
    inputs, labels = sum_mod_p_data(K, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
