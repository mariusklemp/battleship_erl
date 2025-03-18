import torch
import torch.nn as nn
from neat_system.cnn_genome import CNNConvGene, CNNFCGene, CNNPoolGene


def activation_function(activation: str):
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
            board_size=5,
            activation="relu",
            device="cpu",
            genome=None,
            config=None,
            layer_config=None
    ):
        super(ANET, self).__init__()
        self.device = device
        self.input_channels = 5

        if genome is not None and config is not None:
            self.board_size = config.genome_config.input_size
            self.output_size = config.genome_config.output_size
            self._build_from_genome(genome, config)
        elif layer_config is not None:
            self.board_size = board_size
            self.activation_func = activation_function(activation)
            self.output_size = board_size * board_size
            self._build_from_layer_config(layer_config)
        else:
            self.board_size = board_size
            self.activation_func = activation_function(activation)
            self.output_size = board_size * board_size
            self._build_default()

        self.to(device)

    def _build_from_layer_config(self, layer_config):
        self.layer_config = layer_config
        self.logits = nn.Sequential(*self.getLayers())

    def _build_from_genome(self, genome, config):
        input_channels = config.genome_config.input_channels

        self.layers = nn.ModuleList()
        # Track channels, height, and width explicitly.
        curr_c, curr_h, curr_w = input_channels, self.board_size, self.board_size

        for i, gene in enumerate(genome.layer_config):
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                # Use the current channel count as input channels.
                conv_layer = nn.Conv2d(
                    in_channels=curr_c,
                    out_channels=int(gene.out_channels),
                    kernel_size=int(gene.kernel_size),
                    stride=int(gene.stride),
                    padding=int(gene.padding)
                )
                self.layers.append(conv_layer)
                self.layers.append(activation_function(gene.activation))
                # Update the channel count and spatial dimensions.
                curr_c = gene.out_channels
                curr_h = (curr_h + 2 * gene.padding - gene.kernel_size) // gene.stride + 1
                curr_w = (curr_w + 2 * gene.padding - gene.kernel_size) // gene.stride + 1

            elif isinstance(gene, CNNPoolGene):
                if gene.pool_type == "max":
                    pool_layer = nn.MaxPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                else:
                    pool_layer = nn.AvgPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                self.layers.append(pool_layer)
                # Update spatial dimensions based on the pooling operation.
                curr_h = (curr_h - gene.pool_size) // gene.stride + 1
                curr_w = (curr_w - gene.pool_size) // gene.stride + 1

            elif isinstance(gene, CNNFCGene):
                # When hitting the first fully connected layer, flatten the current tensor.
                # Check if the last added layer is not already a Flatten.
                if not self.layers or not isinstance(self.layers[-1], nn.Flatten):
                    self.layers.append(nn.Flatten())
                    fc_input_size = curr_c * curr_h * curr_w
                else:
                    # If flattening was already done, assume gene.input_size is precomputed.
                    fc_input_size = gene.input_size
                fc_layer = nn.Linear(in_features=int(fc_input_size), out_features=int(gene.fc_layer_size))
                self.layers.append(fc_layer)
                self.layers.append(activation_function(gene.activation))
                # After an FC layer, the data is flat.
                curr_c, curr_h, curr_w = gene.fc_layer_size, 1, 1

        # Ensure the final tensor is flattened.
        if not self.layers or not isinstance(self.layers[-1], nn.Flatten):
            self.layers.append(nn.Flatten())
            flattened_size = curr_c * curr_h * curr_w
        else:
            flattened_size = curr_c * curr_h * curr_w

        self.fc_policy = nn.Linear(int(flattened_size), self.output_size)
        self.layers.append(self.fc_policy)

    def _build_default(self):
        self.conv1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc_board = nn.Linear(128 * self.board_size * self.board_size, 1024)
        self.fc_policy = nn.Linear(1024, self.output_size)
        self.fc_value = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.apply(self.init_weights)

    def extra_features_to_board(self, extra_features):
        """
        Converts an extra features vector into a 2D binary board.
        For each index i, it sets the first extra_features[i] rows in column i to 1.
        """
        # Ensure extra_features is a tensor.
        if not torch.is_tensor(extra_features):
            extra_features = torch.tensor(extra_features, dtype=torch.float32)

        board = torch.zeros(self.board_size, self.board_size)

        # Only process up to board_size features
        for i, count in enumerate(extra_features[: self.board_size + 1]):
            count = int(count)  # convert to integer count
            # Clip count to board_size if needed.
            count = min(count, self.board_size)
            # Set the first `count` rows in column i to 1.
            board[:count, i] = 1
        return board

    def forward(self, game_state: torch.Tensor, extra_features):
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
        board_extra = (
            board_extra.unsqueeze(0)
            .unsqueeze(0)
            .repeat(game_state.shape[0], 1, 1, 1)
            .to(self.device)
        )
        # Concatenate the extra channel to the board input along the channel dimension.
        game_state = torch.cat([game_state, board_extra], dim=1)

        # Forward pass through the network
        if hasattr(self, "layers"):
            # NEAT-based architecture
            x = game_state
            for layer in self.layers:
                x = layer(x)
            return x
        elif hasattr(self, "logits"):
            # Layer config based architecture
            return self.logits(game_state)
        else:
            # Default architecture
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
                        layer["in_features"] * self.board_size ** 2,
                        layer["out_features"],
                    )
                else:
                    return nn.Linear(layer["in_features"], layer["out_features"])
            case "Dropout":
                return nn.Dropout(p=layer["p"])
            case "activation":
                return self.activation_func
            case "BatchNorm2d":
                return nn.BatchNorm2d(layer["num_features"])
            case "BatchNorm1d":
                return nn.BatchNorm1d(layer["num_features"])
            case "MaxPool2d":
                return nn.MaxPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", None),
                    padding=layer.get("padding", 0),
                )
            case "AvgPool2d":
                return nn.AvgPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", None),
                    padding=layer.get("padding", 0),
                )
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
