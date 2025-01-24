import torch.nn as nn
import torch.nn.functional as F
import torch
import neat


class DeepNEATCNN(nn.Module):
    def __init__(self, genome, board_size, config):
        super().__init__()
        self.board_size = board_size
        self.config = config
        self.input_channels = (
            4  # Fixed input channels for hits, misses, sunk, and unknown
        )
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.activation_function = self.get_activation_function(
            config.genome_config.activation_default
        )
        self.build_network(genome)

        # print("Convolutional Layers:")
        # print(self.conv_layers)
        # print("Fully Connected Layers:")
        # print(self.fc_layers)

    def get_activation_function(self, activation_name):
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def build_network(self, genome):
        current_size = self.board_size  # Starting size of the board
        input_channels = self.input_channels
        in_features = None

        for layer in genome.layer_config:
            # print("Layer: ", layer)
            if layer["layer_type"] == "conv":
                out_channels = layer["out_channels"]
                kernel_size = min(layer["kernel_size"], current_size)
                stride = layer["stride"]
                padding = layer["padding"]

                # Add Conv layer
                conv_layer = nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

                self.conv_layers.append(conv_layer)
                self.conv_layers.append(self.activation_function)

                input_channels = out_channels
                current_size = self.calculate_conv_output_size(
                    current_size, kernel_size, stride, padding
                )

            elif layer["layer_type"] == "fc":
                if in_features is None:
                    # Transition from Conv to FC layers
                    self.conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    self.fc_layers.append(nn.Flatten())
                    in_features = input_channels

                out_features = layer["fc_layer_size"]
                fc_layer = nn.Linear(in_features, out_features)

                self.fc_layers.append(fc_layer)
                self.fc_layers.append(self.activation_function)
                in_features = out_features

        if in_features is None:
            self.conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc_layers.append(nn.Flatten())
            in_features = input_channels

        self.fc_layers.append(nn.Linear(in_features, self.board_size**2))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, 1, self.board_size, self.board_size)

    @staticmethod
    def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
        output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
        if output_size <= 0:
            raise ValueError(f"Invalid output size: {output_size}")
        return output_size
