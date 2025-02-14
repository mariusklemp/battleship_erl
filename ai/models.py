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
            extra_input_size=5,  # signal that extra features exist
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

        # We now always use 5 input channels: the original 4 plus 1 extra.
        input_channels = 5

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

        self.to(device)

    def extra_features_to_board(self, extra_features):
        """
        Converts an extra features vector into a 2D binary board.
        For each index i, it sets the first extra_features[i] rows in column i to 1.
        """
        # Ensure extra_features is a tensor.
        if not torch.is_tensor(extra_features):
            extra_features = torch.tensor(extra_features, dtype=torch.float32)

        board = torch.zeros(self.board_size, self.board_size)
        for i, count in enumerate(extra_features):
            count = int(count)  # convert to integer count
            # Clip count to board_size if needed.
            count = min(count, self.board_size)
            # Set the first `count` rows in column i to 1.
            board[:count, i] = 1
        return board

    def forward(self, game_state: torch.Tensor, extra_features=None):
        """
        Parameters:
          game_state: Tensor of shape (batch, 4, board_size, board_size)
          extra_features: Tensor or list with extra information (e.g., [0, 2, 1, 0, 0])
                          indicating how many ships remain.
        Returns:
          policy: Tensor of shape (batch, output_size)
        """
        # Ensure game_state is on the correct device.
        game_state = game_state.to(self.device)

        # Convert extra_features to a board (of shape [board_size, board_size])
        board_extra = self.extra_features_to_board(extra_features)
        board_extra = board_extra.unsqueeze(0).unsqueeze(0).repeat(game_state.shape[0], 1, 1, 1).to(self.device)

        # Concatenate the extra channel to the board input along the channel dimension.
        game_state = torch.cat([game_state, board_extra], dim=1)

        print(game_state)

        if hasattr(self, "logits"):
            policy = self.logits(game_state)
        else:
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
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
