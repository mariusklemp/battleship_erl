import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import visualize
from strategies.search.strategy import Strategy


class NNSearch(nn.Module, Strategy):
    def __init__(self, search_agent, net, optimizer="adam", lr=0.001):
        super().__init__()
        self.name = "nn_search"
        self.search_agent = search_agent
        self.net = net
        self.device = net.device
        self.lr = lr
        self.optimizer_name = optimizer

        self.criterion = nn.CrossEntropyLoss()

        # Check if net has parameters — if so, init optimizer now
        self.optimizer = self.get_optimizer()

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
        return board_tensor.view(-1, 5, board_size, board_size).to(self.device)

    def _convert_target(self, target):
        """
        Convert target (assumed to be a probability distribution)
        to a float tensor and add a batch dimension if needed.
        """
        target_tensor = torch.tensor(target, dtype=torch.float32)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        return target_tensor.to(self.device)

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            return optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optimizer_name == "rmsprop":
            return optim.RMSprop(self.net.parameters(), lr=self.lr)
        elif self.optimizer_name == "adagrad":
            return optim.Adagrad(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer")

    def _apply_illegal_mask(self, output, board_tensor):
        # Create a mask matching the output shape.
        illegal_mask = board_tensor[0, 0].reshape(-1) == 1  # shape: [board_size^2]
        # Unsqueeze to match the batch dimension: now shape becomes [1, board_size^2]
        illegal_mask = illegal_mask.unsqueeze(0)
        # Use masked_fill to return a new tensor, without modifying 'output' in-place.
        output = output.masked_fill(illegal_mask, float("-inf"))
        return output

    def find_move(self, state, topp=False):
        # Get both board tensor and extra features from state
        board_tensor = state.state_tensor()

        # Reshape board tensor to (batch, 4, board_size, board_size)
        board_tensor = self._reshape_board(board_tensor)

        # Forward pass to get raw output (logits)
        output = self.net(board_tensor).view(1, -1)
        output = output.to(self.device)

        # Apply illegal move mask via helper function.
        output = self._apply_illegal_mask(output, board_tensor)

        # Apply softmax to convert logits to a probability distribution
        probabilities = nn.functional.softmax(output, dim=-1).squeeze(0)
        probabilities_np = probabilities.detach().numpy()

        # visualize.plot_action_distribution(probabilities_np, self.search_agent.board_size)

        # Choose a move based on the probability distribution
        if topp:
            # argmax of the distribution
            move = np.argmax(probabilities_np)
        else:
            # During training, use pure random sampling
            move = np.random.choice(self.search_agent.board_size ** 2, p=probabilities_np)
        return move, probabilities_np

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
        
        # Prepare batch data
        states = []
        targets = []
        
        for state, target in training_data:
            states.append(state)
            targets.append(target)
        
        # Combine into single batch
        batch_states = torch.stack(states)
        batch_targets = torch.stack([self._convert_target(target).squeeze(0) for target in targets])
        
        # Process entire batch at once
        batch_states = self._reshape_board(batch_states)
        
        self.optimizer.zero_grad()
        outputs = self.net(batch_states)
        
        # Apply illegal move mask for each sample in batch
        for i in range(len(outputs)):
            outputs[i] = self._apply_illegal_mask(outputs[i].unsqueeze(0), batch_states[i].unsqueeze(0)).squeeze(0)
        
        # Add some noise to prevent getting stuck
        if self.training:
            noise = torch.randn_like(outputs) * 0.01
            outputs = outputs + noise
        
        loss = self.criterion(outputs, batch_targets.argmax(dim=1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Record metrics
        self.avg_error_history.append(loss.item())
        
        # Calculate accuracy
        top1_acc, top3_acc = self.calculate_accuracy(outputs, batch_targets)
        self.top1_accuracy_history.append(top1_acc.item())
        self.top3_accuracy_history.append(top3_acc.item())

    def validate_model(self, validation_data):
        self.net.eval()  # Set model to evaluation mode
        
        # Prepare batch data
        states = []
        targets = []
        
        for state, target in validation_data:
            states.append(state)
            targets.append(target)
        
        # Combine into single batch
        batch_states = torch.stack(states)
        batch_targets = torch.stack([self._convert_target(target).squeeze(0) for target in targets])
        
        # Process entire batch at once
        batch_states = self._reshape_board(batch_states)
        
        with torch.no_grad():  # No need to track gradients for validation
            outputs = self.net(batch_states)
            
            # Apply illegal move mask for each sample in batch
            for i in range(len(outputs)):
                outputs[i] = self._apply_illegal_mask(outputs[i].unsqueeze(0), batch_states[i].unsqueeze(0)).squeeze(0)
            
            loss = self.criterion(outputs, batch_targets.argmax(dim=1))
            
            # Calculate accuracies
            top1_acc, top3_acc = self.calculate_accuracy(outputs, batch_targets)
            self.val_top1_accuracy_history.append(top1_acc.item())
            self.val_top3_accuracy_history.append(top3_acc.item())
        
        # Store validation loss
        self.avg_validation_history.append(loss.item())

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
                i * batches_per_epoch: (i + 1) * batches_per_epoch
                ]
            )
            for i in range(len(self.avg_error_history))
        ]
        avg_train_top3 = [
            np.mean(
                self.top3_accuracy_history[
                i * batches_per_epoch: (i + 1) * batches_per_epoch
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
                i * batches_per_epoch_val: (i + 1) * batches_per_epoch_val
                ]
            )
            for i in range(len(self.avg_validation_history))
        ]
        avg_val_top3 = [
            np.mean(
                self.val_top3_accuracy_history[
                i * batches_per_epoch_val: (i + 1) * batches_per_epoch_val
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
        # Save the model
        torch.save(self.net, path)

    def load_model(self, path):
        print(f"Loading model from {path}")
        # Load new state
        self.net = torch.load(path)
        self.net.eval()
