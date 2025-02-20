import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import visualize
from RBUF import RBUF
from ai.models import ANET
from strategies.search.strategy import Strategy

# Note: We use a local import to avoid circular dependencies.
from game_logic.search_agent import SearchAgent


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
        self.device = net.device  # Get device from the network

        # Ensure the network is on the correct device
        self.net = self.net.to(self.device)

        self.optimizer = get_optimizer(optimizer, self.net, lr)
        self.criterion = nn.CrossEntropyLoss()

        self.avg_error_history = []
        self.avg_validation_history = []
        self.top1_accuracy_history = []
        self.top3_accuracy_history = []
        self.val_top1_accuracy_history = []
        self.val_top3_accuracy_history = []

    # ===== Helper Functions =====

    def _reshape_board(self, board_tensor):
        """Reshape board_tensor to (batch, 4, board_size, board_size)."""
        board_size = int(board_tensor.shape[-1])
        return board_tensor.view(-1, 4, board_size, board_size).to(self.device)

    def _convert_target(self, target):
        """
        Convert target (assumed to be a probability distribution)
        to a float tensor and add a batch dimension if needed.
        """
        target_tensor = torch.tensor(target, dtype=torch.float32)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        return target_tensor.to(self.device)

    def _apply_illegal_mask(self, output, board_tensor):
        """
        Mask the output logits so that positions corresponding to illegal moves
        (as indicated in the first channel of board_tensor) are set to -inf.
        """
        illegal_mask = board_tensor[0, 0].reshape(-1) == 1
        output[0, illegal_mask] = float("-inf")
        return output

    def find_move(self, state, topp=False):
        # Get both board tensor and extra features from state
        board_tensor, extra_features = state.state_tensor()

        # Reshape board tensor to (batch, 4, board_size, board_size)
        board_tensor = self._reshape_board(board_tensor)

        # Forward pass to get raw output (logits)
        output = self.net(board_tensor, extra_features).view(1, -1)
        output = output.to(self.device)

        # Mask the output based on the 'unknown' layer (first channel in state.board)
        unknown_layer = torch.tensor(state.board[0], dtype=torch.float32).view(1, -1)
        output[unknown_layer == 1] = float("-inf")

        # Apply softmax to convert logits to a probability distribution
        temperature = 0.5 if topp else 1.0  # Lower temperature for tournament mode
        probabilities = nn.functional.softmax(output / temperature, dim=-1).squeeze(0)
        probabilities_np = probabilities.detach().numpy()

        # Choose a move based on the probability distribution
        if topp:
            # Sample from the distribution with temperature
            # This will still favor high probability moves but allow some exploration
            move = np.random.choice(self.search_agent.board_size**2, p=probabilities_np)
        else:
            # During training, use pure random sampling
            move = np.random.choice(self.search_agent.board_size**2, p=probabilities_np)
        return move

    def calculate_accuracy(self, predicted_values, target_values):
        predicted_probs = nn.functional.softmax(predicted_values, dim=1)
        top1_pred = predicted_probs.argmax(dim=1)
        top3_pred = predicted_probs.topk(
            k=min(3, predicted_probs.size(1)), dim=1
        ).indices

        # If target_values is already 1D (class indices), use it directly.
        if target_values.dim() == 1:
            true_moves = target_values
        else:
            true_moves = target_values.argmax(dim=1)

        top1_accuracy = (top1_pred == true_moves).float().mean()
        top3_accuracy = (
            torch.any(top3_pred == true_moves.unsqueeze(1), dim=1).float().mean()
        )
        return top1_accuracy, top3_accuracy

    def train_model(self, training_data):
        self.net.train()
        total_loss = 0
        total_batches = len(training_data)

        # Track unique moves for diversity monitoring
        all_moves = []
        all_predictions = []

        for batch_idx, (state, target) in enumerate(training_data):
            board_tensor, extra_features = state
            board_tensor = self._reshape_board(board_tensor)
            target_tensor = self._convert_target(target)

            self.optimizer.zero_grad()
            output = self.net(board_tensor, extra_features)
            output = self._apply_illegal_mask(output, board_tensor)

            # Track predictions and targets for diversity analysis
            pred_moves = output.argmax(dim=1).cpu().numpy()
            target_moves = target_tensor.argmax(dim=1).cpu().numpy()
            all_moves.extend(target_moves)
            all_predictions.extend(pred_moves)

            # Add some noise to prevent getting stuck
            if self.training:
                noise = torch.randn_like(output) * 0.01
                output = output + noise

            loss = self.criterion(output, target_tensor.argmax(dim=1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()

            top1_acc, top3_acc = self.calculate_accuracy(output, target_tensor)
            self.top1_accuracy_history.append(top1_acc.item())
            self.top3_accuracy_history.append(top3_acc.item())

        avg_loss = total_loss / total_batches
        self.avg_error_history.append(avg_loss)

    def validate_model(self, validation_data):
        self.net.eval()  # Set model to evaluation mode
        total_loss = 0
        total_batches = len(validation_data)

        with torch.no_grad():  # No need to track gradients for validation
            for state, target in validation_data:
                # Unpack state into board tensor and extra features
                board_tensor, extra_features = state
                board_tensor = self._reshape_board(board_tensor)
                target_tensor = self._convert_target(target)

                # Forward pass
                output = self.net(board_tensor, extra_features)

                # Apply illegal move masking
                output = self._apply_illegal_mask(output, board_tensor)

                # Compute loss using CrossEntropyLoss
                loss = self.criterion(output, target_tensor.argmax(dim=1))

                total_loss += loss.item()

                # Calculate accuracies
                top1_acc, top3_acc = self.calculate_accuracy(output, target_tensor)
                self.val_top1_accuracy_history.append(top1_acc.item())
                self.val_top3_accuracy_history.append(top3_acc.item())

        # Store average validation loss
        avg_val_loss = total_loss / total_batches
        self.avg_validation_history.append(avg_val_loss)

    def plot_metrics(self):
        # Convert loss tensors (or numbers) to NumPy arrays.
        train_losses = np.array(
            [
                x.detach().cpu().item() if torch.is_tensor(x) else x
                for x in self.avg_error_history
            ]
        )
        val_losses = np.array(
            [
                x.detach().cpu().item() if torch.is_tensor(x) else x
                for x in self.avg_validation_history
            ]
        )
        epochs = np.arange(1, len(train_losses) + 1)

        # Group batch-level accuracies into epoch-level averages.
        num_train_batches = len(self.top1_accuracy_history)
        batches_per_epoch = (
            num_train_batches // len(self.avg_error_history)
            if self.avg_error_history
            else num_train_batches
        )

        avg_train_top1 = [
            np.mean(
                self.top1_accuracy_history[
                    i * batches_per_epoch : (i + 1) * batches_per_epoch
                ]
            )
            for i in range(len(self.avg_error_history))
        ]
        avg_train_top3 = [
            np.mean(
                self.top3_accuracy_history[
                    i * batches_per_epoch : (i + 1) * batches_per_epoch
                ]
            )
            for i in range(len(self.avg_error_history))
        ]

        num_val_batches = len(self.val_top1_accuracy_history)
        batches_per_epoch_val = (
            num_val_batches // len(self.avg_validation_history)
            if self.avg_validation_history
            else num_val_batches
        )
        avg_val_top1 = [
            np.mean(
                self.val_top1_accuracy_history[
                    i * batches_per_epoch_val : (i + 1) * batches_per_epoch_val
                ]
            )
            for i in range(len(self.avg_validation_history))
        ]
        avg_val_top3 = [
            np.mean(
                self.val_top3_accuracy_history[
                    i * batches_per_epoch_val : (i + 1) * batches_per_epoch_val
                ]
            )
            for i in range(len(self.avg_validation_history))
        ]

        # Create the plots.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", alpha=0.7)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", alpha=0.7)
        ax1.set_title("Loss Evolution", pad=10)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Soft Cross-Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(epochs, avg_train_top1, "b-", label="Top-1 Train", alpha=0.7)
        ax2.plot(epochs, avg_train_top3, "b--", label="Top-3 Train", alpha=0.7)
        ax2.plot(epochs, avg_val_top1, "r-", label="Top-1 Val", alpha=0.7)
        ax2.plot(epochs, avg_val_top3, "r--", label="Top-3 Val", alpha=0.7)
        ax2.set_title("Accuracy Evolution", pad=10)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        print(f"Saving model to {path}")
        # Save the model
        torch.save(self.net.state_dict(), path)

        # Verify the save by loading and comparing
        saved_state = torch.load(path)
        current_state = self.net.state_dict()

        # Compare a few parameters to ensure they're different from previous saves
        for key in list(current_state.keys())[:3]:  # Check first 3 layers
            print(f"Layer {key} mean value: {current_state[key].mean().item():.6f}")

    def load_model(self, path):
        print(f"Loading model from {path}")
        # Load new state
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
