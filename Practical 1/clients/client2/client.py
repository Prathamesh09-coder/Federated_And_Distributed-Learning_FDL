import copy
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

# Allow importing from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.model import SimpleNN


class Client:
    def __init__(self, client_id, train_loader, device="cpu"):
        self.client_id = client_id
        self.device = device

        self.model = SimpleNN().to(device)
        self.train_loader = train_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, global_weights, epochs=1):
        """
        Train locally using the global model weights.
        """

        # Load the latest global model
        self.model.load_state_dict(copy.deepcopy(global_weights))

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0

            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            print(
                f"Client {self.client_id} | "
                f"Epoch {epoch+1} | "
                f"Loss: {avg_loss:.4f}"
            )

        return copy.deepcopy(self.model.state_dict())