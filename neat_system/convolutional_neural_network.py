import torch
import torch.nn as nn
from cnn_genome import CNNConvGene, CNNFCGene, CNNPoolGene


def compute_conv_output_dim(in_dim, kernel_size, stride, padding):
    """
    Computes the spatial dimension after a convolution layer,
    using the formula: out_dim = (in_dim + 2*padding - kernel_size)//stride + 1.
    """
    return (in_dim + 2 * padding - kernel_size) // stride + 1


def compute_pool_output_dim(in_dim, pool_size, stride):
    """
    Computes the spatial dimension after a pooling layer,
    using the formula: out_dim = (in_dim - pool_size)//stride + 1.
    """
    return (in_dim - pool_size) // stride + 1


class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, output_shape, layer_evals):
        """
        Initialize the Convolutional Neural Network.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_evals = layer_evals

    @staticmethod
    def create(genome, config):
        board_size = config.genome_config.input_size
        input_channels = config.genome_config.input_channels
        input_shape = (input_channels, board_size, board_size)
        output_shape = (board_size, board_size)

        # We'll track the current spatial dimensions as we add layers.
        curr_c, curr_h, curr_w = input_channels, board_size, board_size
        layer_evals = []

        for gene in genome.layer_config:
            # Skip any disabled genes.
            if hasattr(gene, "enabled") and not gene.enabled:
                continue

            if isinstance(gene, CNNConvGene):
                # Check output dimension before building the layer.
                out_h = compute_conv_output_dim(curr_h, gene.kernel_size, gene.stride, gene.padding)
                out_w = compute_conv_output_dim(curr_w, gene.kernel_size, gene.stride, gene.padding)
                if out_h < 1 or out_w < 1:
                    raise ValueError(
                        f"Invalid Conv layer: kernel={gene.kernel_size}, stride={gene.stride}, "
                        f"padding={gene.padding} would produce spatial dims ({out_h}x{out_w})."
                    )
                conv_layer = nn.Conv2d(
                    in_channels=int(gene.in_channels),
                    out_channels=int(gene.out_channels),
                    kernel_size=int(gene.kernel_size),
                    stride=int(gene.stride),
                    padding=int(gene.padding),
                )
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    conv_layer.weight.data.copy_(torch.from_numpy(gene.weights))
                    conv_layer.bias.data.copy_(torch.from_numpy(gene.biases))
                activation = ConvolutionalNeuralNetwork.get_activation_function(gene.activation)
                layer_evals.append({"type": "conv", "params": {"layer": conv_layer, "activation": activation}})
                curr_c = gene.out_channels
                curr_h, curr_w = out_h, out_w

            elif isinstance(gene, CNNPoolGene):
                out_h = compute_pool_output_dim(curr_h, gene.pool_size, gene.stride)
                out_w = compute_pool_output_dim(curr_w, gene.pool_size, gene.stride)
                if out_h < 1 or out_w < 1:
                    raise ValueError(
                        f"Invalid Pool layer: pool_size={gene.pool_size}, stride={gene.stride} "
                        f"would produce spatial dims ({out_h}x{out_w})."
                    )
                if gene.pool_type == "max":
                    pool_layer = nn.MaxPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                else:
                    pool_layer = nn.AvgPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride))
                layer_evals.append({"type": "pool", "params": {"layer": pool_layer}})
                curr_h, curr_w = out_h, out_w

            elif isinstance(gene, CNNFCGene):
                activation = ConvolutionalNeuralNetwork.get_activation_function(gene.activation)
                fc_layer = nn.Linear(in_features=int(gene.input_size), out_features=int(gene.fc_layer_size))
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    fc_layer.weight.data.copy_(torch.from_numpy(gene.weights))
                    fc_layer.bias.data.copy_(torch.from_numpy(gene.biases))
                layer_evals.append({"type": "fc", "params": {"layer": fc_layer, "activation": activation}})
            else:
                raise ValueError(f"Unknown gene type: {type(gene)}")

        # Check for at least one FC layer.
        last_fc_size = None
        for gene in reversed(genome.layer_config):
            if isinstance(gene, CNNFCGene) and getattr(gene, "enabled", True):
                last_fc_size = gene.fc_layer_size
                break
        if last_fc_size is None:
            raise ValueError("No fully connected (FC) layer found in the genome.")

        output_layer = nn.Linear(int(last_fc_size), board_size ** 2)
        layer_evals.append({"type": "fc", "params": {"layer": output_layer, "activation": nn.Identity()}})

        return ConvolutionalNeuralNetwork(input_shape, output_shape, layer_evals)

    def activate(self, inputs):
        """
        Perform a forward pass through the network.
        Args:
            inputs: A tensor representing the input data.
        Returns:
            Output tensor after passing through the network.
        """
        x = inputs

        for i, layer in enumerate(self.layer_evals):
            layer_type, layer_params = layer["type"], layer["params"]

            if layer_type == "conv":
                conv_layer = layer_params["layer"]
                x = conv_layer(x)
                x = layer_params["activation"](x)

            elif layer_type == "pool":
                pool_layer = layer_params["layer"]
                x = pool_layer(x)  # Apply max/avg pooling (no activation function)

            elif layer_type == "fc":
                x = x.view(x.size(0), -1)  # Flatten before FC layers
                fc_layer = layer_params["layer"]
                x = fc_layer(x)
                x = layer_params["activation"](x)

        # Final reshape to match the board size.
        output_shape = (-1, *self.output_shape)
        return x.view(*output_shape)  # Reshape to the board size

    @staticmethod
    def get_activation_function(name):
        if name.lower() == "relu":
            return nn.ReLU()
        elif name.lower() == "sigmoid":
            return nn.Sigmoid()
        elif name.lower() == "tanh":
            return nn.Tanh()
        else:
            # Default to Identity if unknown.
            return nn.Identity()
