import torch
import torch.nn as nn
from cnn_genome import CNNConvGene, CNNFCGene, CNNPoolGene


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
        """
        Construct a Convolutional Neural Network phenotype from a genome.
        """
        board_size = config.genome_config.input_size
        input_shape = (config.genome_config.input_channels, board_size, board_size)
        output_shape = (board_size, board_size)
        layer_evals = []  # Holds layer specifications

        # Iterate over the gene objects in the genome.
        for gene in genome.layer_config:
            if isinstance(gene, CNNConvGene):
                activation = ConvolutionalNeuralNetwork.get_activation_function(gene.activation)

                # Create the convolutional layer.
                conv_layer = nn.Conv2d(
                    in_channels=int(gene.in_channels),
                    out_channels=int(gene.out_channels),
                    kernel_size=int(gene.kernel_size),
                    stride=int(gene.stride),
                    padding=int(gene.padding),
                )
                # Copy evolved weights and biases into the layer.
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    conv_layer.weight.data.copy_(torch.from_numpy(gene.weights).type_as(conv_layer.weight.data))
                    conv_layer.bias.data.copy_(torch.from_numpy(gene.biases).type_as(conv_layer.bias.data))

                layer_evals.append({"type": "conv", "params": {"layer": conv_layer, "activation": activation}})

            elif isinstance(gene, CNNPoolGene):
                # Create a pooling layer (Max or Average)
                pool_layer = nn.MaxPool2d(kernel_size=int(gene.pool_size), stride=int(gene.stride)) \
                    if gene.pool_type == "max" else nn.AvgPool2d(kernel_size=int(gene.pool_size),
                                                                 stride=int(gene.stride))

                layer_evals.append({"type": "pool", "params": {"layer": pool_layer}})

            elif isinstance(gene, CNNFCGene):
                activation = ConvolutionalNeuralNetwork.get_activation_function(gene.activation)

                # Create the fully connected (Linear) layer.
                fc_layer = nn.Linear(
                    in_features=int(gene.input_size),
                    out_features=int(gene.fc_layer_size),
                )
                # Copy evolved weights and biases into the layer.
                if hasattr(gene, "weights") and hasattr(gene, "biases"):
                    fc_layer.weight.data.copy_(torch.from_numpy(gene.weights).type_as(fc_layer.weight.data))
                    fc_layer.bias.data.copy_(torch.from_numpy(gene.biases).type_as(fc_layer.bias.data))

                layer_evals.append({"type": "fc", "params": {"layer": fc_layer, "activation": activation}})

        # Assume the last FC gene defines the size for the output layer.
        last_fc_size = None
        for gene in reversed(genome.layer_config):
            if isinstance(gene, CNNFCGene):
                last_fc_size = gene.fc_layer_size
                break

        if last_fc_size is None:
            raise ValueError("No fully connected (fc) layer found in the genome.")

        # Create the output layer.
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
