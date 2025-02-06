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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.training_losses = []
        self.validation_losses = []
        self.top1_accuracy_history = []
        self.top3_accuracy_history = []
        self.val_top1_accuracy_history = []
        self.val_top3_accuracy_history = []

    def find_move(self, state):
        self.search_agent.move_count += 1

        # Get both board tensor and extra features
        board_tensor, extra_features = state.state_tensor()

        # Reshape the board to have shape (batch_size, channels, height, width)
        board_size = self.search_agent.board_size
        board_tensor = board_tensor.view(1, 4, board_size, board_size)
        extra_features = extra_features.unsqueeze(0)  # Add batch dimension

        # Forward pass to get raw output (logits)
        output = self.net(board_tensor, extra_features).view(
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

    def calculate_accuracy(self, predicted_values, target_values):
        """
        Calculate accuracy for move prediction.

        Args:
            predicted_values: Model output logits
            target_values: True target moves

        Returns:
            tuple: (accuracy, top_3_accuracy)
        """
        # Apply softmax to get probabilities
        predicted_probs = F.softmax(predicted_values, dim=1)

        # Get top-k predictions
        top1_pred = predicted_probs.argmax(dim=1)
        top3_pred = predicted_probs.topk(
            k=min(3, predicted_probs.size(1)), dim=1
        ).indices

        # Get true moves (assuming target is one-hot encoded)
        true_moves = target_values.argmax(dim=1)

        # Calculate accuracies
        top1_accuracy = (top1_pred == true_moves).float().mean()
        top3_accuracy = (
            torch.any(top3_pred == true_moves.unsqueeze(1), dim=1).float().mean()
        )

        return top1_accuracy, top3_accuracy

    def train(self, training_data, validation_data):
        """
        Train the actor network on the value output.
        Each data point includes a game state tensor and a target value.
        :param data: list of tuples, (state_tensor, target_value)
        """
        self.net.train()
        states, target_values = zip(*training_data)

        # Unpack the state tuples into board tensors and extra features
        board_tensors, extra_features = zip(*[state for state in states])

        # Stack states and reshape to [batch_size, channels, height, width]
        board_tensor = torch.stack(board_tensors)
        board_tensor = board_tensor.squeeze(1)  # Remove the extra dimension
        board_size = int(board_tensor.shape[-1] ** 0.5)
        board_tensor = board_tensor.view(-1, 4, board_size, board_size)

        # Stack extra features
        extra_tensor = torch.stack(extra_features)  # Shape: [batch_size, 6]

        # Convert the target values from a list of ndarrays to a single ndarray
        target_values = np.array(target_values)
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32).squeeze(
            1
        )

        self.optimizer.zero_grad()

        # Forward pass to get the value predictions
        predicted_values = self.net(board_tensor, extra_tensor)

        # Compute the cross entropy loss using nn.CrossEntropyLoss
        policy_loss = nn.CrossEntropyLoss()(predicted_values, target_values_tensor)

        # Backward pass and optimize
        policy_loss.backward()
        self.optimizer.step()

        # Calculate accuracies
        top1_acc, top3_acc = self.calculate_accuracy(
            predicted_values, target_values_tensor
        )
        # print(f"{self.name} value loss: {policy_loss.item()}")
        # print(f"Training Accuracy - Top-1: {top1_acc:.3f}, Top-3: {top3_acc:.3f}")

        self.training_losses.append(policy_loss.item())
        self.top1_accuracy_history.append(top1_acc.item())
        self.top3_accuracy_history.append(top3_acc.item())

        # Validate the model
        if validation_data:
            val_loss = self.validate(validation_data)
            self.scheduler.step(val_loss)

    def validate(self, validation_data):
        """
        Validate the actor network on the value output.
        Each data point includes a game state tensor and a target value.
        :param data: list of tuples, (state_tensor, target_value)
        """
        with torch.no_grad():
            self.net.eval()
            states, target_values = zip(*validation_data)

            # Unpack the state tuples into board tensors and extra features
            board_tensors, extra_features = zip(*[state for state in states])

            # Stack states and reshape to [batch_size, channels, height, width]
            board_tensor = torch.stack(board_tensors)
            board_tensor = board_tensor.squeeze(1)
            board_size = int(board_tensor.shape[-1] ** 0.5)
            board_tensor = board_tensor.view(-1, 4, board_size, board_size)

            # Stack extra features
            extra_tensor = torch.stack(extra_features)

            # Convert the target values from a list of ndarrays to a single ndarray
            target_values = np.array(target_values)
            target_values_tensor = torch.tensor(
                target_values, dtype=torch.float32
            ).squeeze(1)

            # Forward pass to get the value predictions
            predicted_values = self.net(board_tensor, extra_tensor)
            policy_loss = nn.CrossEntropyLoss()(predicted_values, target_values_tensor)

            # Calculate accuracies
            top1_acc, top3_acc = self.calculate_accuracy(
                predicted_values, target_values_tensor
            )

            self.validation_losses.append(policy_loss.item())
            self.val_top1_accuracy_history.append(top1_acc.item())
            self.val_top3_accuracy_history.append(top3_acc.item())

            return policy_loss.item()

    def plot_metrics(self):
        """Plot training metrics with improved visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot losses
        ax1.plot(self.training_losses, "b-", label="Training Loss", alpha=0.7)
        ax1.plot(self.validation_losses, "r-", label="Validation Loss", alpha=0.7)
        ax1.set_title("Loss Evolution", pad=10)
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Plot accuracies
        ax2.plot(self.top1_accuracy_history, "b-", label="Top-1 Train", alpha=0.7)
        ax2.plot(self.top3_accuracy_history, "b--", label="Top-3 Train", alpha=0.7)
        ax2.plot(self.val_top1_accuracy_history, "r-", label="Top-1 Val", alpha=0.7)
        ax2.plot(self.val_top3_accuracy_history, "r--", label="Top-3 Val", alpha=0.7)
        ax2.set_title("Accuracy Evolution", pad=10)
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # Add summary statistics
        if len(self.training_losses) > 0:
            stats_text = (
                f"Final Metrics:\n"
                f"Train Loss: {self.training_losses[-1]:.3f}\n"
                f"Val Loss: {self.validation_losses[-1]:.3f}\n"
                f"Top-1 Train: {self.top1_accuracy_history[-1]:.3f}\n"
                f"Top-3 Train: {self.top3_accuracy_history[-1]:.3f}\n"
                f"Top-1 Val: {self.val_top1_accuracy_history[-1]:.3f}\n"
                f"Top-3 Val: {self.val_top3_accuracy_history[-1]:.3f}"
            )
            fig.text(
                0.98,
                0.98,
                stats_text,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        self.net.save(path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
