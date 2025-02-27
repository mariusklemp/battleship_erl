import torch
import torch.nn as nn
from neat_system.cnn_genome import CNNConvGene, CNNFCGene, CNNPoolGene


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
        output_size=None,
        activation="relu",
        device="cpu",
        layer_config=None,
        extra_input_size=5,  # signal that extra features exist
        genome=None,
        config=None,
    ):
        """
        Parameters:
          board_size: Dimension of the square board (e.g., 5 for a 5x5 board)
          output_size: Number of outputs (e.g., 25 for a 5x5 board move distribution)
          activation: String specifying the activation function
          device: Torch device to use
          layer_config: Optional JSON-like configuration for the layers
          extra_input_size: Size of extra input features
          genome: Optional genome for NEAT-based architecture
          config: Optional configuration for NEAT-based architecture
        """
        super(ANET, self).__init__()
        self.board_size = board_size
        self.activation_func = activation_function(activation)
        self.device = device

        # Set output size based on board size if not provided
        if output_size is None:
            self.output_size = board_size * board_size
        else:
            self.output_size = output_size

        # We now always use 5 input channels: the original 4 plus 1 extra.
        self.input_channels = 5

        if genome is not None and config is not None:
            # Create network from genome (NEAT approach)
            self.create_from_genome(genome, config)
        elif layer_config is not None:
            # Create network from layer configuration
            self.layer_config = layer_config
            self.logits = nn.Sequential(*self.getLayers())
        else:
            # Default architecture
            # Board branch: process board with CNN layers.
            self.conv1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.fc_board = nn.Linear(128 * board_size * board_size, 1024)
            # Final layers: policy and value heads.
            self.fc_policy = nn.Linear(1024, self.output_size)
            self.fc_value = nn.Linear(1024, 1)
            self.dropout = nn.Dropout(p=0.3)
            self.apply(self.init_weights)

        self.to(device)

    def create_from_genome(self, genome, config):
        """
        Create a neural network from a NEAT genome.

        Parameters:
            genome: The NEAT genome
            config: The NEAT configuration
        """
        print(f"\nCreating ANET from genome {genome.key}")

        # Get board size from config
        board_size = config.genome_config.input_size

        # Create layers list to hold all network components
        self.layers = nn.ModuleList()

        # Track the current spatial size and channels
        current_size = board_size

        # Flag to track if we've added a flatten layer
        flattened = False
        last_conv_out_channels = None
        flattened_size = None

        # Process each gene in the genome's layer configuration
        for i, gene in enumerate(genome.layer_config):
            print(f"Processing gene: {gene}")

            if isinstance(gene, CNNConvGene):  # Conv layer
                # Create convolutional layer with adjusted input channels for the first layer
                if i == 0:
                    # First layer should accept 5 channels instead of 4
                    conv_layer = nn.Conv2d(
                        in_channels=self.input_channels,  # Use 5 channels
                        out_channels=int(gene.out_channels),
                        kernel_size=int(gene.kernel_size),
                        stride=int(gene.stride),
                        padding=int(gene.padding),
                    )
                else:
                    # Subsequent layers use the gene's in_channels
                    conv_layer = nn.Conv2d(
                        in_channels=int(gene.in_channels),
                        out_channels=int(gene.out_channels),
                        kernel_size=int(gene.kernel_size),
                        stride=int(gene.stride),
                        padding=int(gene.padding),
                    )

                # Copy weights if available (and adjust for the first layer)
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    if i == 0:
                        # For the first layer, we need to adjust the weights to handle 5 channels
                        # We'll use the first 4 channels as is and initialize the 5th channel with zeros
                        original_weights = torch.from_numpy(gene.weights)
                        new_weights = torch.zeros(
                            (
                                gene.out_channels,
                                self.input_channels,
                                gene.kernel_size,
                                gene.kernel_size,
                            ),
                            dtype=original_weights.dtype,
                        )
                        # Copy the original weights for the first 4 channels
                        new_weights[:, :5, :, :] = original_weights
                        conv_layer.weight.data.copy_(
                            new_weights.type_as(conv_layer.weight.data)
                        )
                    else:
                        conv_layer.weight.data.copy_(
                            torch.from_numpy(gene.weights).type_as(
                                conv_layer.weight.data
                            )
                        )

                    conv_layer.bias.data.copy_(
                        torch.from_numpy(gene.biases).type_as(conv_layer.bias.data)
                    )

                # Add layer and activation
                self.layers.append(conv_layer)
                self.layers.append(self.activation_func)

                # Update tracking variables
                in_channels = gene.out_channels
                last_conv_out_channels = gene.out_channels
                current_size = (
                    (current_size + 2 * gene.padding - gene.kernel_size) // gene.stride
                ) + 1

            elif isinstance(gene, CNNPoolGene):  # Pooling layer
                # Create pooling layer
                if gene.pool_type == "max":
                    pool_layer = nn.MaxPool2d(
                        kernel_size=int(gene.pool_size), stride=int(gene.stride)
                    )
                else:
                    pool_layer = nn.AvgPool2d(
                        kernel_size=int(gene.pool_size), stride=int(gene.stride)
                    )

                # Add layer
                self.layers.append(pool_layer)

                # Update spatial size
                current_size = ((current_size - gene.pool_size) // gene.stride) + 1

            elif isinstance(gene, CNNFCGene):  # Fully connected layer
                # If this is the first FC layer, add a flatten layer
                if not flattened:
                    self.layers.append(nn.Flatten())
                    flattened = True

                    # Calculate flattened size
                    flattened_size = (
                        current_size * current_size * last_conv_out_channels
                    )
                    fc_input_size = flattened_size
                else:
                    fc_input_size = gene.input_size

                # Create FC layer
                fc_layer = nn.Linear(
                    in_features=int(fc_input_size), out_features=int(gene.fc_layer_size)
                )

                # Copy weights if available
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    fc_layer.weight.data.copy_(
                        torch.from_numpy(gene.weights).type_as(fc_layer.weight.data)
                    )
                    fc_layer.bias.data.copy_(
                        torch.from_numpy(gene.biases).type_as(fc_layer.bias.data)
                    )

                # Add layer and activation
                self.layers.append(fc_layer)
                self.layers.append(self.activation_func)

                # Update flattened size for next layer
                flattened_size = gene.fc_layer_size

        # If we haven't flattened yet, do it now
        if not flattened and last_conv_out_channels is not None:
            self.layers.append(nn.Flatten())
            flattened_size = current_size * current_size * last_conv_out_channels
        elif not flattened:
            # If there were no conv layers, flatten the input directly
            self.layers.append(nn.Flatten())
            flattened_size = board_size * board_size * self.input_channels

        # Add final output layer
        self.fc_policy = nn.Linear(int(flattened_size), self.output_size)
        self.layers.append(self.fc_policy)

        print(f"Created network with {len(self.layers)} layers")
        print(
            f"Final flattened size: {flattened_size}, Output size: {self.output_size}"
        )

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

        # Process extra features if provided
        if extra_features is not None:
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
        else:
            # If no extra features provided and the input has only 4 channels, create a zero tensor
            if game_state.shape[1] == 4:  # Only add if we have exactly 4 channels
                board_size = game_state.shape[
                    2
                ]  # Assuming game_state is [batch, channels, height, width]
                zero_channel = torch.zeros(
                    game_state.shape[0], 1, board_size, board_size, device=self.device
                )
                game_state = torch.cat([game_state, zero_channel], dim=1)

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
                        layer["in_features"] * self.board_size**2,
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

    @classmethod
    def create_from_cnn_genome(cls, genome, config, device="cpu"):
        """
        Factory method to create an ANET instance from a CNN genome.

        Parameters:
            genome: The CNN genome
            config: The NEAT configuration
            device: The device to use

        Returns:
            An ANET instance
        """
        board_size = config.genome_config.input_size
        output_size = config.genome_config.output_size

        return cls(
            board_size=board_size,
            output_size=output_size,
            device=device,
            genome=genome,
            config=config,
        )

    def activate(self, input_tensor):
        """
        Compatibility method for NEAT_search strategy.
        This is an alias for the forward method without extra_features.

        Parameters:
            input_tensor: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor
        """
        # Print debug information about the input tensor
        print(f"Input tensor shape in activate: {input_tensor.shape}")

        # Check if the input tensor has 4 channels (NEAT_search strategy)
        if input_tensor.shape[1] == 4:
            # Add a dummy extra channel filled with zeros
            batch_size, _, height, width = input_tensor.shape

            # Ensure height and width match the expected board size
            if height != self.board_size or width != self.board_size:
                print(
                    f"WARNING: Input tensor spatial dimensions ({height}x{width}) don't match board size ({self.board_size}x{self.board_size})"
                )
                # Resize the tensor to match the expected board size
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor,
                    size=(self.board_size, self.board_size),
                    mode="nearest",
                )

            extra_channel = torch.zeros(
                (batch_size, 1, self.board_size, self.board_size),
                device=input_tensor.device,
            )
            input_tensor = torch.cat([input_tensor, extra_channel], dim=1)

        # Forward pass
        output = self.forward(input_tensor, extra_features=None)

        return output
