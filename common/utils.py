import copy
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_datasets(num_clients=3, batch_size=32):
    """
    Load MNIST and split it equally among clients.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Split training data equally
    client_size = len(train_dataset) // num_clients

    lengths = [client_size] * num_clients

    # Add remaining samples to last client
    lengths[-1] += len(train_dataset) - sum(lengths)

    client_datasets = random_split(train_dataset, lengths)

    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_datasets
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return client_loaders, test_loader


def fed_avg(client_weights):
    """
    Federated Averaging
    """

    avg_weights = copy.deepcopy(client_weights[0])

    for key in avg_weights.keys():

        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]

        avg_weights[key] = avg_weights[key] / len(client_weights)

    return avg_weights