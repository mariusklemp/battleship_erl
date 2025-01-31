import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super(ANET, self).__init__()
        self.board_size = board_size
        self.activation_func = activation_function(activation)
        self.device = device
        self.output_size = output_size

        input_channels = 4

        if layer_config is None:
            # Shared layers
            self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()

            # Policy head
            self.fc_shared = nn.Linear(128 * board_size * board_size, 1024)
            self.fc_policy = nn.Linear(1024, output_size)

            # Value head
            self.fc_value = nn.Linear(1024, 1)

            self.apply(self.init_weights)

            self.dropout = nn.Dropout(p=0.3)
        else:
            self.layer_config = layer_config
            self.logits = nn.Sequential(*self.getLayers())

    def forward(self, game_state: torch.Tensor):
        game_state = game_state.to(self.device)
        x = game_state
        if hasattr(self, "logits"):
            policy = self.logits(x)
        else:
            model = nn.Sequential(
                self.conv1,
                self.activation_func,
                self.conv2,
                self.activation_func,
                self.flatten,
                self.fc_shared,
                self.activation_func,
                self.dropout,
            )
            policy = self.fc_policy(model(x))

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
                if layer["last"]:
                    return nn.Linear(layer["in_features"], self.output_size)
                elif layer["dynamic"]:
                    return nn.Linear(
                        layer["in_features"] * self.board_size * self.board_size,
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
        """
        Initializes weights to be zero
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
