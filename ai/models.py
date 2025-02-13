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
            extra_input_size,
            device=torch.device("cpu"),
            layer_config=None,
    ):
        super(ANET, self).__init__()
        self.board_size = board_size
        self.activation_func = activation_function(activation)
        self.device = device
        self.output_size = output_size
        self.extra_input_size = extra_input_size

        # Early fusion: Increase input channels by extra_input_size.
        # Board originally has 4 channels, so new input channels = 4 + extra_input_size.
        input_channels = 4 + self.extra_input_size

        if layer_config is None:
            # Shared layers with early-fused input
            self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()

            # Policy head (and value head if needed)
            self.fc_shared = nn.Linear(128 * board_size * board_size, 1024)
            self.fc_policy = nn.Linear(1024, output_size)
            self.fc_value = nn.Linear(1024, 1)

            self.dropout = nn.Dropout(p=0.3)
            self.apply(self.init_weights)
        else:
            self.layer_config = layer_config
            self.logits = nn.Sequential(*self.getLayers())

    def forward(self, game_state: torch.Tensor, extra_input: torch.Tensor):
        """
        Parameters:
          game_state: Tensor of shape (batch, 4, board_size, board_size)
          extra_input: Tensor of shape (batch, extra_input_size) e.g., [0, 1, 1, 1, 0]
        """
        game_state = game_state.to(self.device)
        extra_input = extra_input.to(self.device)

        # Expand extra_input spatially:
        # extra_input: (batch, extra_input_size) -> (batch, extra_input_size, 1, 1)
        # Then expand to match board spatial dimensions (board_size x board_size)
        batch, _, h, w = game_state.size()
        extra_expanded = extra_input.unsqueeze(-1).unsqueeze(-1).expand(batch, self.extra_input_size, h, w)

        # Early fusion: Concatenate along the channel dimension
        fused_input = torch.cat([game_state, extra_expanded], dim=1)  # (batch, 4+extra_input_size, h, w)

        if hasattr(self, "logits"):
            policy = self.logits(fused_input)
        else:
            # Process through convolutional layers
            x = self.conv1(fused_input)
            x = self.activation_func(x)
            x = self.conv2(x)
            x = self.activation_func(x)
            x = self.flatten(x)

            board_features = self.fc_shared(x)
            board_features = self.activation_func(board_features)
            board_features = self.dropout(board_features)

            # Final policy layer (and value head if used)
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
