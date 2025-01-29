import torch.nn as nn


class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, output_shape, layer_evals):
        """
        Initialize the Convolutional Neural Network.
        """
        self.input_shape = input_shape  # Input shape: (channels, height, width)
        self.output_shape = output_shape  # Output shape: (board_size, board_size)
        self.layer_evals = layer_evals  # Layer definitions (type, parameters, etc.)

    @staticmethod
    def create(genome, config):
        """
        Construct a Convolutional Neural Network phenotype from a genome.
        """
        board_size = config.genome_config.input_size
        input_shape = (
            config.genome_config.input_channels,
            board_size,
            board_size,
        )  # 4 input channels
        output_shape = (board_size, board_size)  # Board size
        layer_evals = []  # Holds layer specifications

        for layer_config in genome.layer_config:
            activation = ConvolutionalNeuralNetwork.get_activation_function(
                layer_config["activation"]
            )

            if layer_config["layer_type"] == "conv":
                conv_layer = nn.Conv2d(
                    in_channels=layer_config["in_channels"],  # Use genome's in_channels
                    out_channels=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"],
                )
                layer_evals.append(
                    {
                        "type": "conv",
                        "params": {"layer": conv_layer, "activation": activation},
                    }
                )

            elif layer_config["layer_type"] == "fc":
                fc_layer = nn.Linear(
                    in_features=layer_config["input_size"],
                    out_features=layer_config["fc_layer_size"],
                )
                layer_evals.append(
                    {
                        "type": "fc",
                        "params": {"layer": fc_layer, "activation": activation},
                    }
                )

        # Final output layer
        output_layer = nn.Linear(layer_config["fc_layer_size"], board_size**2)
        layer_evals.append(
            {
                "type": "fc",
                "params": {"layer": output_layer, "activation": nn.Identity()},
            }
        )

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

            elif layer_type == "fc":
                x = x.view(x.size(0), -1)
                fc_layer = layer_params["layer"]
                x = fc_layer(x)
                x = layer_params["activation"](x)

        # Final reshape to match the board size
        output_shape = (-1, *self.output_shape)

        return x.view(*output_shape)  # Reshape to the board size

    @staticmethod
    def get_activation_function(name):
        """Retrieve the activation function by name."""
        if name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
