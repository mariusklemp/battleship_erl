import torch.nn as nn
import random


class DeepNEATCNN(nn.Module):
    def __init__(self, genome, config, board_size):
        super().__init__()
        self.config = config
        self.board_size = board_size
        self.conv_layers = nn.Sequential()  # Sequential container for convolutional layers
        self.fc_layers = nn.Sequential()  # Sequential container for fully connected layers
        self.build_network(genome)

    def build_network(self, genome):
        # Start with 4 input channels for the 4 board layers
        input_channels = 4
        current_size = self.board_size

        # Use genome to determine structure of convolutional layers
        for idx, node in enumerate(genome.nodes.values()):
            if getattr(node, 'type', None) == 'conv' and random.random() < self.config.conv_layer_add_prob:
                # Select parameters from the config
                out_channels = random.choice(self.config.conv_channels)
                kernel_size = random.choice(self.config.conv_filter_sizes)
                stride = random.choice(self.config.conv_strides)
                padding = (kernel_size - 1) // 2  # To maintain spatial size

                # Add convolutional layer
                self.conv_layers.append(nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ))
                self.conv_layers.append(nn.ReLU())

                # Update parameters for next layer
                input_channels = out_channels
                current_size = self.calculate_conv_output_size(current_size, kernel_size, stride, padding)

            elif getattr(node, 'type', None) == 'fc' and random.random() < self.config.fc_layer_add_prob:
                # Add pooling after convolution layers for consistent flatten size
                self.conv_layers.append(nn.AdaptiveAvgPool2d((current_size, current_size)))

                # Flatten and create fully connected layer based on the convolutional output size
                self.fc_layers.append(nn.Flatten())
                in_features = input_channels * current_size * current_size
                out_features = random.choice(self.config.fc_layer_sizes)

                # Add fully connected layer
                self.fc_layers.append(nn.Linear(in_features, out_features))
                self.fc_layers.append(nn.ReLU())
                in_features = out_features  # Update for next FC layer

        # Ensure in_features is defined based on the final convolutional output
        if 'in_features' not in locals():
            self.conv_layers.append(nn.AdaptiveAvgPool2d((current_size, current_size)))
            in_features = input_channels * current_size * current_size
            self.fc_layers.append(nn.Flatten())

        # Final output layer to match the board size
        self.fc_layers.append(nn.Linear(in_features, self.board_size ** 2))

    def forward(self, board_tensor):
        # Pass input through convolutional layers
        x = self.conv_layers(board_tensor)

        # Then pass through fully connected layers
        x = self.fc_layers(x)
        return x

    @staticmethod
    def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1
