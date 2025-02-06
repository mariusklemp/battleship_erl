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

        # Extra input branch: process ship information vector
        # For example, we assume extra_input_size is the dimension of the extra features.
        self.fc_extra = nn.Sequential(
            nn.Linear(self.extra_input_size, 128),
            self.activation_func,
            nn.Linear(128, 128),
            self.activation_func,
        )

        # After processing, we will combine the board branch (1024-dim) with extra branch (128-dim)
        combined_dim = 1024 + 128
        # Final fully connected layer for policy (you can also add an extra hidden layer if desired)
        self.fc_policy = nn.Linear(combined_dim, output_size)

        # Optionally, you might also add a value head:
        self.fc_value = nn.Linear(combined_dim, 1)

        self.apply(self.init_weights)

    def forward(self, game_state: torch.Tensor, extra_input: torch.Tensor):
        # game_state: (batch, 4, board_size, board_size)
        # extra_input: (batch, extra_input_size)

        game_state = game_state.to(self.device)
        extra_input = extra_input.to(self.device)

        if hasattr(self, "logits"):
            # If using layer_config, use that branch (not shown here)
            policy = self.logits(game_state)
        else:
            # Process board through convolutional layers
            x = self.conv1(game_state)
            x = self.activation_func(x)
            x = self.conv2(x)
            x = self.activation_func(x)
            x = self.flatten(x)
            board_features = self.fc_shared(x)
            board_features = self.activation_func(board_features)
            board_features = self.dropout(board_features)

            # Process extra input through extra branch
            extra_features = self.fc_extra(extra_input)
            # Concatenate along the feature (last) dimension
            combined = torch.cat([board_features, extra_features], dim=1)

            # Pass through final policy layer
            policy = self.fc_policy(combined)
            # Optionally, if you need a value head as well:
            # value = self.fc_value(combined)
            # return policy, value

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
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)  # He initialization
            nn.init.constant_(m.bias, 0)

    def save(self, path: str):
        torch.save(self.state_dict(), path)


# # Example input
# board_size = 5
# ship_sizes = [5, 4, 3, 3, 2]
#
# board = [[0 for _ in range(board_size**2)] for _ in range(4)]
#
# # Simulated CNN Input: (batch, 4, board_size, board_size)
# board_input = torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)
#
#
#
# # Initialize model
# model = ANET(board_size=board_size, output_size=output_size, activation="relu", max_ships=5)
#
# # Forward pass
# output = model(board_input, ship_sizes_input)
# print(output.shape)  # Expected: (batch_size, board_size * board_size)
