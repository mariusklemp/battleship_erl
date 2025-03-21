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
        # Initialize the module list and the spatial/channel counters.
        self.layers = nn.ModuleList()

        curr_h, curr_w, curr_c = self.board_size, self.board_size, config.genome_config.input_channels
        layer_evals = []

        # Iterate over the genome's layer configuration.
        for gene in genome.layer_config:
            # Skip any disabled genes.
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                # Compute expected output dimensions.
                out_h = (curr_h + 2 * gene.padding - gene.kernel_size) // gene.stride + 1
                out_w = (curr_w + 2 * gene.padding - gene.kernel_size) // gene.stride + 1

                # Create the convolution layer.
                conv_layer = nn.Conv2d(
                    in_channels=int(gene.in_channels),
                    out_channels=int(gene.out_channels),
                    kernel_size=int(gene.kernel_size),
                    stride=int(gene.stride),
                    padding=int(gene.padding)
                )
                # Copy over the weights and biases if provided.
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    conv_layer.weight.data.copy_(torch.from_numpy(gene.weights))
                    conv_layer.bias.data.copy_(torch.from_numpy(gene.biases))
                activation = activation_function(gene.activation)
                layer_evals.append({"type": "conv", "params": {"layer": conv_layer, "activation": activation}})
                curr_h, curr_w = out_h, out_w
                curr_c = gene.out_channels

            elif isinstance(gene, CNNPoolGene):
                out_h = (curr_h - gene.pool_size) // gene.stride + 1
                out_w = (curr_w - gene.pool_size) // gene.stride + 1

                if gene.pool_type == "max":
                    pool_layer = nn.MaxPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                else:
                    pool_layer = nn.AvgPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                layer_evals.append({"type": "pool", "params": {"layer": pool_layer}})
                curr_h, curr_w = out_h, out_w

            elif isinstance(gene, CNNFCGene):
                activation = activation_function(gene.activation)
                fc_layer = nn.Linear(int(gene.input_size), int(gene.fc_layer_size))
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    fc_layer.weight.data.copy_(torch.from_numpy(gene.weights))
                    fc_layer.bias.data.copy_(torch.from_numpy(gene.biases))
                layer_evals.append({"type": "fc", "params": {"layer": fc_layer, "activation": activation}})
            else:
                raise ValueError(f"Unknown gene type: {type(gene)}")

        # Determine the last FC gene to define the final output layer.
        last_fc_size = None
        for gene in reversed(genome.layer_config):
            if isinstance(gene, CNNFCGene) and getattr(gene, "enabled", True):
                last_fc_size = gene.fc_layer_size
                break

        if last_fc_size is None:
            fc_input_size = curr_c * curr_h * curr_w
            last_fc_size = fc_input_size

        output_layer = nn.Linear(int(last_fc_size), self.output_size)
        layer_evals.append({"type": "fc", "params": {"layer": output_layer, "activation": nn.Identity()}})

        # Convert the layer evaluation list into a ModuleList.
        for item in layer_evals:
            ltype = item["type"]
            params = item["params"]
            if ltype in ["conv"]:
                self.layers.append(params["layer"])
                self.layers.append(params["activation"])
            elif ltype in ["pool"]:
                self.layers.append(params["layer"])
            elif ltype in ["fc"]:
                flatten_layer = nn.Flatten()
                self.layers.append(flatten_layer)
                self.layers.append(params["layer"])
                self.layers.append(params["activation"])

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
        Handles the ship matrix representation.
        The input is already a binary matrix where each column represents a ship size
        and each 1 in that column indicates a ship.
        
        Args:
            extra_features: Binary matrix of shape (batch_size, board_size, board_size) where:
                          - Each column i represents ships of size i+1
                          - Each 1 in column i indicates a ship of size i+1
            
        Returns:
            The same matrix, ensuring it's on the correct device
        """
        # Ensure extra_features is a tensor and has the correct shape
        if not torch.is_tensor(extra_features):
            extra_features = torch.tensor(extra_features, dtype=torch.float32)
            
        # Ensure the matrix has the correct shape (including batch dimension)
        if extra_features.shape[1:] != (self.board_size, self.board_size):
            raise ValueError(f"Expected shape (batch_size, {self.board_size}, {self.board_size}), got {extra_features.shape}")
            
        return extra_features

    def forward(self, game_state: torch.Tensor, extra_features):
        # Ensure game_state is on the correct device.
        game_state = game_state.to(self.device)

        # Convert extra_features to a board (of shape [board_size, board_size])
        board_extra = self.extra_features_to_board(extra_features)
        
        # Add channel dimension and repeat to match game_state batch size
        board_extra = (
            board_extra.unsqueeze(1)  # Add channel dimension
            .repeat(game_state.shape[0], 1, 1, 1)  # Repeat to match batch size
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
