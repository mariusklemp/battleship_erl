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
    def __init__(self, search_agent, net, optimizer="adam", lr=0.001, name="nn_search"):
        super().__init__()
        self.name = name
        self.search_agent = search_agent
        self.net = net
        self.device = net.device  # Get device from the network

        # Ensure the network is on the correct device
        self.net = self.net.to(self.device)

        self.optimizer = get_optimizer(optimizer, self.net, lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
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
        output = output.cpu()

        # Mask the output based on the 'unknown' layer (first channel in state.board)
        unknown_layer = torch.tensor(state.board[0], dtype=torch.float32).view(1, -1)
        output[unknown_layer == 1] = float("-inf")

        # Apply softmax to convert logits to a probability distribution
        probabilities = nn.functional.softmax(output, dim=-1).squeeze(0)
        probabilities_np = probabilities.detach().numpy()
        print("Probabilities:", probabilities_np)

        # Choose a move based on the probability distribution
        if topp:
            visualize.plot_action_distribution(
                probabilities_np, self.search_agent.board_size
            )
            move = np.argmax(probabilities_np)
        else:
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

        error_history = []
        for batch_idx, (state, target) in enumerate(training_data):
            print(f"Training batch {batch_idx + 1}")

            # Unpack state into board tensor and extra features.
            board_tensor, extra_features = state
            board_tensor = self._reshape_board(board_tensor)

            # Forward pass: get network output (raw logits)
            output = self.net(board_tensor, extra_features)
            output = self._apply_illegal_mask(output, board_tensor)

            # Compute predicted probabilities using softmax.
            probabilities = F.softmax(output, dim=-1)
            print("Output after masking:", probabilities)

            # Convert target to a tensor.
            target_tensor = self._convert_target(target)
            print("Target tensor:", target_tensor)

            # Compute loss using soft cross-entropy.
            # TODO Check if this is the correct loss function.
            loss = -(target_tensor * torch.log(probabilities + 1e-8)).sum(dim=-1).mean()
            print("Loss:", loss)
            error_history.append(loss)

            # Update parameters: zero gradients, backpropagate, and update weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate and store accuracies.
            top1_acc, top3_acc = self.calculate_accuracy(output, target_tensor)
            self.top1_accuracy_history.append(top1_acc.item())
            self.top3_accuracy_history.append(top3_acc.item())

        # Store the average loss over the training batches.
        self.avg_error_history.append(sum(error_history) / len(error_history))

    def validate_model(self, validation_data):
        self.net.eval()  # Set model to evaluation mode
        error_history = []
        with torch.no_grad():
            for state, target in validation_data:
                board_tensor, extra_features = state
                board_tensor = self._reshape_board(board_tensor)

                # Forward pass: get network output (raw logits)
                output = self.net(board_tensor, extra_features)
                output = self._apply_illegal_mask(output, board_tensor)

                # Compute predicted probabilities.
                probabilities = F.softmax(output, dim=-1)

                # Convert target to a tensor.
                target_tensor = self._convert_target(target)

                # Compute soft cross-entropy loss.
                # TODO Check if this is the correct loss function.
                loss = -(target_tensor * torch.log(probabilities + 1e-8)).sum(dim=-1).mean()

                error_history.append(loss)

                # Calculate and store accuracies.
                top1_acc, top3_acc = self.calculate_accuracy(output, target_tensor)
                self.val_top1_accuracy_history.append(top1_acc.item())
                self.val_top3_accuracy_history.append(top3_acc.item())

            # Store the average validation loss over all batches.
            self.avg_validation_history.append(sum(error_history) / len(error_history))

    def plot_metrics(self):
        # Convert loss tensors (or numbers) to NumPy arrays.
        train_losses = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for x in self.avg_error_history])
        val_losses = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for x in self.avg_validation_history])
        epochs = np.arange(1, len(train_losses) + 1)

        # Group batch-level accuracies into epoch-level averages.
        num_train_batches = len(self.top1_accuracy_history)
        batches_per_epoch = num_train_batches // len(self.avg_error_history) if self.avg_error_history else num_train_batches

        avg_train_top1 = [np.mean(self.top1_accuracy_history[i * batches_per_epoch:(i + 1) * batches_per_epoch])
                          for i in range(len(self.avg_error_history))]
        avg_train_top3 = [np.mean(self.top3_accuracy_history[i * batches_per_epoch:(i + 1) * batches_per_epoch])
                          for i in range(len(self.avg_error_history))]

        num_val_batches = len(self.val_top1_accuracy_history)
        batches_per_epoch_val = num_val_batches // len(self.avg_validation_history) if self.avg_validation_history else num_val_batches
        avg_val_top1 = [np.mean(self.val_top1_accuracy_history[i * batches_per_epoch_val:(i + 1) * batches_per_epoch_val])
                        for i in range(len(self.avg_validation_history))]
        avg_val_top3 = [np.mean(self.val_top3_accuracy_history[i * batches_per_epoch_val:(i + 1) * batches_per_epoch_val])
                        for i in range(len(self.avg_validation_history))]

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
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()


# === Example usage ===
if __name__ == "__main__":
    # Load configuration.
    layer_config = json.load(open("../../ai/config.json"))
    board_size = 5
    net = ANET(
        board_size=board_size,
        activation="relu",
        output_size=board_size ** 2,
        device="cpu",
        layer_config=layer_config,
    )

    # Create the search agent.
    search_agent = SearchAgent(
        board_size=board_size,
        strategy="nn_search",
        net=net,
        optimizer="adam",
        lr=0.001,
    )

    # Initialize the replay buffer and load data.
    rbuf = RBUF(max_len=10000)
    rbuf.load_from_file(file_path="../../rbuf/rbuf.pkl")
    print("Loaded replay buffer from file")

    # Retrieve a training set and a validation set.
    validation_set = rbuf.get_validation_set()
    print("Length validation set:", len(validation_set))

    # Train for multiple iterations, training on each batch multiple times.
    for epoch in range(200):
        training_set = rbuf.get_training_set(batch_size=20)
        print(f"Epoch {epoch + 1}")
        search_agent.strategy.train_model(training_set)
        search_agent.strategy.validate_model(validation_set)

    # Plot training and validation metrics.
    search_agent.strategy.plot_metrics()
    search_agent.strategy.save_model(f"../../models/model.pth")
