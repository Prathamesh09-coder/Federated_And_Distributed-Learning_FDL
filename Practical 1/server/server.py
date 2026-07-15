import copy
import sys
import os

import torch

# Allow importing from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.model import SimpleNN
from common.utils import load_datasets, fed_avg
from clients.client1.client import Client


def evaluate(model, test_loader, device="cpu"):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using Device: {device}")

    # Global Model
    global_model = SimpleNN().to(device)

    # Create client datasets
    client_loaders, test_loader = load_datasets(
        num_clients=3,
        batch_size=32
    )

    # Create Clients
    clients = []

    for i in range(3):
        client = Client(
            client_id=i + 1,
            train_loader=client_loaders[i],
            device=device
        )
        clients.append(client)

    rounds = 5

    for round_num in range(rounds):

        print("\n" + "=" * 40)
        print(f"Round {round_num + 1}")
        print("=" * 40)

        global_weights = copy.deepcopy(global_model.state_dict())

        client_weights = []

        # Send model to every client
        for client in clients:

            updated_weights = client.train(
                global_weights,
                epochs=1
            )

            client_weights.append(updated_weights)

        # Federated Averaging
        averaged_weights = fed_avg(client_weights)

        # Update Global Model
        global_model.load_state_dict(averaged_weights)

        # Evaluate
        accuracy = evaluate(
            global_model,
            test_loader,
            device
        )

        print(f"\nGlobal Accuracy: {accuracy:.2f}%")

    print("\nTraining Finished!")


if __name__ == "__main__":
    main()