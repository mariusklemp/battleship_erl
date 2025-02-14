import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
        self.training_losses = []
        self.validation_losses = []
        self.top1_accuracy_history = []
        self.top3_accuracy_history = []
        self.val_top1_accuracy_history = []
        self.val_top3_accuracy_history = []

    def find_move(self, state, topp=False):
        # Get both board tensor and extra features from state
        board_tensor, extra_features = state.state_tensor()

        # Reshape board tensor to (batch_size, channels, height, width)
        board_size = self.search_agent.board_size
        board_tensor = board_tensor.view(1, 4, board_size, board_size).to(self.device)
        extra_features = extra_features.unsqueeze(0).to(
            self.device
        )  # Add batch dimension

        # Forward pass to get raw output (logits)
        output = self.net(board_tensor).view(1, -1)

        # Move output back to CPU for numpy operations
        output = output.cpu()

        # Mask the output based on the 'unknown' layer (first channel in state.board)
        unknown_layer = torch.tensor(state.board[0], dtype=torch.float32).view(1, -1)
        output[unknown_layer == 1] = float("-inf")

        # Apply softmax to convert logits to a probability distribution
        probabilities = nn.functional.softmax(output, dim=-1).squeeze(0)
        probabilities_np = probabilities.detach().numpy()

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

    def train(self, training_data, validation_data, device=None):
        if device is None:
            device = self.device

        self.net.train()
        states, target_values = zip(*training_data)

        # Unpack each state into board tensor and extra features
        board_tensors, extra_features = zip(*states)

        # Stack and reshape board tensors: (batch, 4, board_size, board_size)
        board_tensor = torch.stack(board_tensors).squeeze(1)
        board_size = int(board_tensor.shape[-1])
        board_tensor = board_tensor.view(-1, 4, board_size, board_size).to(device)

        # Stack extra features and move to device
        extra_tensor = torch.stack(extra_features).to(device)

        # Convert target values to a tensor and move to device
        target_values = np.array(target_values)
        if target_values.ndim > 1 and target_values.shape[1] > 1:
            target_values = target_values.argmax(axis=1)
        target_values_tensor = torch.tensor(target_values, dtype=torch.long).to(device)

        # Zero gradients before forward pass
        self.optimizer.zero_grad()

        try:
            # Forward pass
            predicted_values = self.net(board_tensor)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            policy_loss = criterion(predicted_values, target_values_tensor)

            # Backward pass and optimization
            policy_loss.backward()
            self.optimizer.step()

            # Calculate and store accuracies
            top1_acc, top3_acc = self.calculate_accuracy(
                predicted_values, target_values_tensor
            )
            self.training_losses.append(policy_loss.item())
            self.top1_accuracy_history.append(top1_acc.item())
            self.top3_accuracy_history.append(top3_acc.item())

        except RuntimeError as e:
            if "MPS" in str(e):
                # Fallback to CPU if MPS operation fails
                self.device = torch.device("cpu")
                self.net = self.net.to(self.device)
                return self.train(training_data, validation_data, self.device)
            else:
                raise e

        # Validate if validation_data is provided
        if validation_data:
            val_loss = self.validate(validation_data, device)
            self.scheduler.step(val_loss)

    def validate(self, validation_data, device=None):
        if device is None:
            device = self.device

        self.net.eval()
        with torch.no_grad():
            try:
                states, target_values = zip(*validation_data)
                board_tensors, extra_features = zip(*states)

                # Process board tensors
                board_tensor = torch.stack(board_tensors).squeeze(1)
                board_size = int(board_tensor.shape[-1])
                board_tensor = board_tensor.view(-1, 4, board_size, board_size).to(
                    device
                )

                # Process extra features
                extra_tensor = torch.stack(extra_features).to(device)

                # Process target values
                target_values = np.array(target_values)
                if target_values.ndim > 1 and target_values.shape[1] > 1:
                    target_values = target_values.argmax(axis=1)
                target_values_tensor = torch.tensor(target_values, dtype=torch.long).to(
                    device
                )

                predicted_values = self.net(board_tensor)
                policy_loss = nn.CrossEntropyLoss()(
                    predicted_values, target_values_tensor
                )

                top1_acc, top3_acc = self.calculate_accuracy(
                    predicted_values, target_values_tensor
                )
                self.validation_losses.append(policy_loss.item())
                self.val_top1_accuracy_history.append(top1_acc.item())
                self.val_top3_accuracy_history.append(top3_acc.item())

                return policy_loss.item()

            except RuntimeError as e:
                if "MPS" in str(e):
                    # Fallback to CPU if MPS operation fails
                    self.device = torch.device("cpu")
                    self.net = self.net.to(self.device)
                    return self.validate(validation_data, self.device)
                else:
                    raise e

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(self.training_losses, "b-", label="Training Loss", alpha=0.7)
        ax1.plot(self.validation_losses, "r-", label="Validation Loss", alpha=0.7)
        ax1.set_title("Loss Evolution", pad=10)
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()
        ax2.plot(self.top1_accuracy_history, "b-", label="Top-1 Train", alpha=0.7)
        ax2.plot(self.top3_accuracy_history, "b--", label="Top-3 Train", alpha=0.7)
        ax2.plot(self.val_top1_accuracy_history, "r-", label="Top-1 Val", alpha=0.7)
        ax2.plot(self.val_top3_accuracy_history, "r--", label="Top-3 Val", alpha=0.7)
        ax2.set_title("Accuracy Evolution", pad=10)
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))


# Example usage
if __name__ == "__main__":
    # Load configuration
    layer_config = json.load(open("../../ai/config.json"))

    board_size = 5
    net = ANET(
        board_size=board_size,
        activation="relu",
        output_size=board_size**2,
        device="cpu",
        layer_config=layer_config,
        extra_input_size=5,
    )

    # Create the search agent
    search_agent = SearchAgent(
        board_size=board_size,
        strategy="nn_search",
        net=net,
        optimizer="adam",
        lr=0.001,
    )

    # Initialize the replay buffer and load data
    rbuf = RBUF(max_len=10000)
    rbuf.load_from_file(file_path="../../rbuf/rbuf.pkl")
    print("Loaded replay buffer from file")

    # Retrieve a training set and a validation set
    validation_set = rbuf.get_validation_set()
    print("Length validation set:", len(validation_set))

    # Train for multiple epochs, validating at each epoch
    for epoch in range(100):
        training_set = rbuf.get_training_set(batch_size=200)
        print(f"Epoch {epoch + 1}")
        search_agent.strategy.train(training_set, validation_set)

    # Plot training and validation metrics
    search_agent.strategy.plot_metrics()
