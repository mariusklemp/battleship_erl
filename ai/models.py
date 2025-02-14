import torch
import torch.nn as nn


def activation_function(activation: str):
    """Utility function to return an activation function module."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class ANET(nn.Module):
    def __init__(
        self,
        board_size,
        output_size,
        activation,
        device=torch.device("cpu"),
        layer_config=None,
    ):
        """
        Parameters:
          board_size: Dimension of the square board (e.g., 5 for a 5x5 board)
          output_size: Number of outputs (e.g., 25 for a 5x5 board move distribution)
          activation: String specifying the activation function
          device: Torch device to use
          layer_config: Optional JSON-like configuration for the layers
        """
        super(ANET, self).__init__()
        self.board_size = board_size
        self.activation_func = activation_function(activation)
        self.device = device
        self.output_size = output_size

        # Since extra features are removed, we use only 4 channels for the board.
        input_channels = 4

        if layer_config is None:
            # Board branch: process board with CNN layers.
            self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.fc_board = nn.Linear(128 * board_size * board_size, 1024)

            # Final layers: policy and value heads.
            self.fc_policy = nn.Linear(1024, output_size)
            self.fc_value = nn.Linear(1024, 1)
            self.dropout = nn.Dropout(p=0.3)
            self.apply(self.init_weights)
        else:
            self.layer_config = layer_config
            self.logits = nn.Sequential(*self.getLayers())

        # Move the model to the specified device.
        self.to(device)

    def forward(self, game_state: torch.Tensor):
        """
        Parameters:
          game_state: Tensor of shape (batch, 4, board_size, board_size)
        Returns:
          policy: Tensor of shape (batch, output_size)
        """
        # Ensure the game state is on the correct device.
        game_state = game_state.to(self.device)

        if hasattr(self, "logits"):
            policy = self.logits(game_state)
        else:
            # Process board input through convolutional layers.
            x = self.conv1(game_state)
            x = self.activation_func(x)
            x = self.conv2(x)
            x = self.activation_func(x)
            x = self.flatten(x)
            board_features = self.fc_board(x)
            board_features = self.activation_func(board_features)
            board_features = self.dropout(board_features)
            policy = self.fc_policy(board_features)

        return policy

    def getLayer(self, layer):
        match layer["type"]:
            case "Conv2d":
                return nn.Conv2d(
                    layer["in_channels"],
                    layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                )
            case "Flatten":
                return nn.Flatten()
            case "Linear":
                if layer.get("last", False):
                    return nn.Linear(layer["in_features"], self.output_size)
                elif layer.get("dynamic", False):
                    return nn.Linear(
                        layer["in_features"] * self.board_size**2,
                        layer["out_features"],
                    )
                else:
                    return nn.Linear(layer["in_features"], layer["out_features"])
            case "Dropout":
                return nn.Dropout(p=layer["p"])
            case "activation":
                return self.activation_func
            case _:
                raise ValueError(f"Unsupported layer type: {layer}")

    def getLayers(self):
        layers = []
        for layer in self.layer_config["layers"]:
            layers.append(self.getLayer(layer))
        return layers

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)  # He initialization
            nn.init.constant_(m.bias, 0)

    def save(self, path: str):
        torch.save(self.state_dict(), path)


# Example usage:
if __name__ == "__main__":
    # Suppose we have a batch of 8 examples on a 5x5 board with 4 channels,
    # and 25 output classes.
    batch_size = 8
    board_size = 5
    output_size = board_size * board_size  # 25 possible moves

    net = ANET(
        board_size=board_size,
        output_size=output_size,
        activation="relu",
        extra_input_size=0,
    )
    # Create a dummy board state: (batch, 4, board_size, board_size)
    game_state = torch.randn(batch_size, 4, board_size, board_size)
    # Forward pass.
    logits = net(game_state)
    print("Output logits shape:", logits.shape)  # Expected: [8, 25]
