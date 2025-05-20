import json
import os

import torch
import torch.nn as nn
from neat_system.cnn_genome import CNNConvGene, CNNFCGene, CNNPoolGene


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
        self.genome = genome

        if genome is not None and self.config is not None:
            self.board_size = config.genome_config.input_size
            self.output_size = config.genome_config.output_size
            layer_list = self._build_from_genome(genome)
            self.logits = nn.Sequential(*layer_list)
            self.layers = list(self.logits.children())
            self.read_weights_biases_from_genome(genome)
        elif layer_config:
            with open(layer_config, "r") as f:
                cfg = json.load(f)

            self.activation_func = self.get_activation_function(activation)
            self._build_from_layer_config(cfg)

        self.to(device)

    def _build_from_layer_config(self, layer_config):
        self.layer_config = layer_config
        self.logits = nn.Sequential(*self.getLayers())
        # Initialize all weights
        self.logits.apply(self.init_weights)

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
                layer_list.append(self.get_activation_function(gene.activation))
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
                    layer_list.append(self.get_activation_function(gene.activation))
            else:
                raise ValueError(f"Unknown gene type: {type(gene)}")
        return layer_list


    def forward(self, game_state: torch.Tensor):
        game_state = game_state.to(self.device)
        if hasattr(self, "logits"):
            return self.logits(game_state)

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
            case "GlobalAvgPool2d":
                return nn.AdaptiveAvgPool2d((1, 1))
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

    def get_activation_function(self, activation: str):
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

    def save_model(self, path: str):
        """
        Save just the state_dict of this network.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load_model(self, path: str):
        """
        Load into this network from a state_dict checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        # ckpt might be a plain state_dict or wrapped in a dict
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.load_state_dict(state_dict)
        self.eval()

    def save_model_genome(self, path: str):
        """
        Save both the state_dict and the NEAT genome (and optionally the optimizer).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'genome': self.genome,
        }

        torch.save(checkpoint, path)



