import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from strategies.search.strategy import Strategy


def get_optimizer(optimizer: str, model, lr: float) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer")


class NNSearch(nn.Module, Strategy):
    def __init__(self, search_agent, net, optimizer="adam", lr=0.001):
        super().__init__()
        self.name = "nn_search"
        self.search_agent = search_agent
        self.net = net
        self.optimizer = get_optimizer(optimizer, self.net, lr)
        self.training_losses = []
        self.validation_losses = []
        self.accuracy_history = []

    def find_move(self, state):
        self.search_agent.move_count += 1

        # Convert the board state (4 layers) into a tensor
        board_tensor = torch.tensor(state.board, dtype=torch.float32)

        # Reshape the board to have shape (batch_size, channels, height, width)
        # For example, (1, 4, 10, 10) for a 10x10 board
        board_size = self.search_agent.board_size
        board_tensor = board_tensor.view(1, 4, board_size, board_size)

        # Forward pass to get raw output (logits)
        output = self.net.forward(board_tensor).view(
            1, -1
        )  # Flatten the output to (1, board_size^2)

        # Get the 'unknown' layer (first layer) and flip it to mark known squares
        unknown_layer = torch.tensor(state.board[0], dtype=torch.float32).view(1, -1)

        # Mask the output: Set values where unknown_layer is 1 to -inf
        output[unknown_layer == 1] = float("-inf")

        # Apply softmax to convert to probability distribution
        probabilities = F.softmax(output, dim=-1).squeeze(0)  # Shape: (board_size^2,)

        # Convert tensor to numpy array for random.choice
        probabilities_np = probabilities.detach().numpy()

        # Choose a move based on the probability distribution
        move = np.random.choice(self.search_agent.board_size**2, p=probabilities_np)

        return move

    def train(self, training_data, validation_data):
        """
        Train the actor network on the value output.
        Each data point includes a game state tensor and a target value.
        :param data: list of tuples, (state_tensor, target_value)
        """
        self.net.train()
        states, target_values = zip(*training_data)

        # Stack states and reshape to [batch_size, channels, height, width]
        states_tensor = torch.stack(
            states
        )  # Shape: [batch_size, 1, channels, board_size*board_size]
        states_tensor = states_tensor.squeeze(1)  # Remove the extra dimension
        board_size = int(
            states_tensor.shape[-1] ** 0.5
        )  # Calculate board size from flattened dimension
        states_tensor = states_tensor.view(
            -1, 4, board_size, board_size
        )  # Reshape to proper dimensions

        # Convert the target values from a list of ndarrays to a single ndarray
        target_values = np.array(target_values)
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32).squeeze(
            1
        )

        self.optimizer.zero_grad()

        # Forward pass to get the value predictions
        predicted_values = self.net(states_tensor)

        # Compute the cros entropy loss using nn.CrossEntropyLoss
        policy_loss = nn.CrossEntropyLoss()(predicted_values, target_values_tensor)

        # Backward pass and optimize
        policy_loss.backward()
        self.optimizer.step()

        self.training_losses.append(policy_loss.item())

        print(f"{self.name} value loss: {policy_loss.item()}")

        # Calculate the accuracy of the model
        accuracy = (
            (predicted_values.argmax(dim=1) == target_values_tensor.argmax(dim=1))
            .float()
            .mean()
        )
        print(f"Training Accuracy: {accuracy}")
        self.accuracy_history.append(accuracy.item())

        # Validate the model
        if validation_data:
            self.validate(validation_data)

    def validate(self, validation_data):
        """
        Validate the actor network on the value output.
        Each data point includes a game state tensor and a target value.
        :param data: list of tuples, (state_tensor, target_value)
        """
        with torch.no_grad():
            self.net.eval()
            states, target_values = zip(*validation_data)

            # Stack states and reshape to [batch_size, channels, height, width]
            states_tensor = torch.stack(
                states
            )  # Shape: [batch_size, 1, channels, board_size*board_size]
            states_tensor = states_tensor.squeeze(1)  # Remove the extra dimension
            board_size = int(
                states_tensor.shape[-1] ** 0.5
            )  # Calculate board size from flattened dimension
            states_tensor = states_tensor.view(
                -1, 4, board_size, board_size
            )  # Reshape to proper dimensions

            # Convert the target values from a list of ndarrays to a single ndarray
            target_values = np.array(target_values)
            target_values_tensor = torch.tensor(
                target_values, dtype=torch.float32
            ).squeeze(1)

            # Forward pass to get the value predictions
            predicted_values = self.net(states_tensor)
            policy_loss = nn.CrossEntropyLoss()(predicted_values, target_values_tensor)
            self.validation_losses.append(policy_loss.item())

            # Calculate the accuracy
            predicted_values = F.softmax(predicted_values, dim=1)
            predicted_values = torch.argmax(predicted_values, dim=1)
            correct = (predicted_values == target_values_tensor.argmax(dim=1)).sum()
            total = target_values_tensor.size(0)
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy}")

            print(f"{self.name} validation loss: {policy_loss.item()}")

    def plot_metrics(self):
        # Plot the training and validation losses and accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label="Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.show()
