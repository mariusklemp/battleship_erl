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
            layer_config=None,
    ):
        super(ANET, self).__init__()
        self.device = device
        self.input_channels = 5
        self.config = config
        self.board_size = board_size
        self.output_size = board_size * board_size

        if genome is not None and self.config is not None:
            self.board_size = config.genome_config.input_size
            self.output_size = config.genome_config.output_size
            layer_list = self._build_from_genome(genome)
            self.logits = nn.Sequential(*layer_list)
            self.layers = list(self.logits.children())
            self.read_weights_biases_from_genome(genome)

        elif layer_config is not None:
            self.activation_func = activation_function(activation)
            self._build_from_layer_config(layer_config)

        else:
            self.activation_func = activation_function(activation)
            self._build_default()

        self.to(device)

    def _build_from_layer_config(self, layer_config):
        self.layer_config = layer_config
        self.logits = nn.Sequential(*self.getLayers())

    def _build_from_genome(self, genome):
        layer_list = []
        curr_h, curr_w = self.board_size, self.board_size
        curr_c = self.config.genome_config.input_channels

        # Precompute the last enabled FC gene.
        enabled_fc_genes = [gene for gene in genome.layer_config
                            if isinstance(gene, CNNFCGene) and (not hasattr(gene, "enabled") or gene.enabled)]
        last_fc_gene = enabled_fc_genes[-1] if enabled_fc_genes else None

        # Iterate over the genome's layer configuration.
        for gene in genome.layer_config:
            # Skip disabled genes.
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                conv_layer = nn.Conv2d(
                    in_channels=int(gene.in_channels),
                    out_channels=int(gene.out_channels),
                    kernel_size=int(gene.kernel_size),
                    stride=int(gene.stride),
                    padding=int(gene.padding)
                )
                layer_list.append(conv_layer)
                layer_list.append(activation_function(gene.activation))
                out_h = (curr_h + 2 * gene.padding - gene.kernel_size) // gene.stride + 1
                out_w = (curr_w + 2 * gene.padding - gene.kernel_size) // gene.stride + 1
                curr_h, curr_w = out_h, out_w
                curr_c = gene.out_channels

            elif isinstance(gene, CNNPoolGene):
                pool_layer = (nn.MaxPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                              if gene.pool_type == "max"
                              else nn.AvgPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride)))
                layer_list.append(pool_layer)
                curr_h = (curr_h - gene.pool_size) // gene.stride + 1
                curr_w = (curr_w - gene.pool_size) // gene.stride + 1

            elif isinstance(gene, CNNFCGene):
                if not any(isinstance(layer, nn.Flatten) for layer in layer_list):
                    layer_list.append(nn.Flatten())
                fc_layer = nn.Linear(int(gene.input_size), int(gene.fc_layer_size))
                layer_list.append(fc_layer)
                # Only append activation if this FC gene is not the last enabled FC gene.
                if gene != last_fc_gene:
                    layer_list.append(activation_function(gene.activation))
            else:
                raise ValueError(f"Unknown gene type: {type(gene)}")
        return layer_list


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
        if not torch.is_tensor(extra_features):
            extra_features = torch.tensor(extra_features, dtype=torch.float32)
        if extra_features.shape[1:] != (self.board_size, self.board_size):
            raise ValueError(
                f"Expected shape (batch_size, {self.board_size}, {self.board_size}), got {extra_features.shape}")
        return extra_features

    def forward(self, game_state: torch.Tensor, extra_features):
        game_state = game_state.to(self.device)
        board_extra = self.extra_features_to_board(extra_features)
        board_extra = board_extra.unsqueeze(1).repeat(game_state.shape[0], 1, 1, 1).to(self.device)
        game_state = torch.cat([game_state, board_extra], dim=1)
        if hasattr(self, "logits"):
            return self.logits(game_state)
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

    def read_weights_biases_from_genome(self, genome):
        """
        Reads the weights and biases from the genome and copies them into the model's
        convolutional and fully connected layers.

        Assumes:
          - The model has been built from the genome (using _build_from_genome).
          - The ordering of enabled genome genes in genome.layer_config matches the
            corresponding layers in self.layers (filtered for nn.Conv2d and nn.Linear).
        """
        # Filter out the network's Conv2d and Linear layers.
        net_conv_layers = [layer for layer in self.layers if isinstance(layer, nn.Conv2d)]
        net_fc_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        conv_index = 0
        fc_index = 0

        for gene in genome.layer_config:
            # Only process enabled genes (or genes without an "enabled" attribute)
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                if conv_index >= len(net_conv_layers):
                    raise ValueError("Mismatch between number of conv genes and conv layers in the model.")

                conv_layer = net_conv_layers[conv_index]
                # Copy gene weights and biases into the conv layer using your method.
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    with torch.no_grad():
                        conv_layer.weight.copy_(
                            torch.tensor(gene.weights, device=self.device, dtype=conv_layer.weight.dtype)
                        )
                        conv_layer.bias.copy_(
                            torch.tensor(gene.biases, device=self.device, dtype=conv_layer.bias.dtype)
                        )
                conv_index += 1

            elif isinstance(gene, CNNFCGene):
                if fc_index >= len(net_fc_layers):
                    raise ValueError("Mismatch between number of fc genes and fc layers in the model.")
                fc_layer = net_fc_layers[fc_index]
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    with torch.no_grad():
                        fc_layer.weight.copy_(
                            torch.tensor(gene.weights, device=self.device, dtype=fc_layer.weight.dtype)
                        )
                        fc_layer.bias.copy_(
                            torch.tensor(gene.biases, device=self.device, dtype=fc_layer.bias.dtype)
                        )
                fc_index += 1

    def read_weights_biases_to_genome(self, genome):
        """
        Copies the model's current weights and biases from its convolutional and
        fully connected layers back into the corresponding genome genes.

        Assumes:
          - The model has been built from the genome (using _build_from_genome).
          - The enabled gene ordering in genome.layer_config corresponds to the order
            of trainable layers (nn.Conv2d and nn.Linear) in self.layers.
        """
        net_conv_layers = [layer for layer in self.layers if isinstance(layer, nn.Conv2d)]
        net_fc_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        conv_index = 0
        fc_index = 0

        for gene in genome.layer_config:
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                conv_layer = net_conv_layers[conv_index]
                gene.weights = conv_layer.weight.data.cpu().numpy()
                gene.biases = conv_layer.bias.data.cpu().numpy()
                conv_index += 1

            elif isinstance(gene, CNNFCGene):
                fc_layer = net_fc_layers[fc_index]
                gene.weights = fc_layer.weight.data.cpu().numpy()
                gene.biases = fc_layer.bias.data.cpu().numpy()
                fc_index += 1

        return genome

