import torch.nn as nn


class DeepNEATCNN(nn.Module):
    def __init__(self, genome, board_size):
        super().__init__()
        self.board_size = board_size
        self.input_channels = 4  # Fixed input channels for hits, misses, sunk, and unknown
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.build_network(genome)

        print("Convolutional Layers:")
        print(self.conv_layers)
        print("Fully Connected Layers:")
        print(self.fc_layers)

    def build_network(self, genome):
        current_size = self.board_size  # Starting size of the board
        input_channels = self.input_channels
        in_features = None

        for layer in genome.layer_config:
            print(layer)
            if layer['layer_type'] == 'conv':
                out_channels = layer['out_channels']
                kernel_size = min(layer['kernel_size'], current_size)
                stride = layer['stride']
                padding = layer['padding']

                # Add Conv layer
                self.conv_layers.append(nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ))
                self.conv_layers.append(nn.ReLU())

                input_channels = out_channels
                current_size = self.calculate_conv_output_size(current_size, kernel_size, stride, padding)

            elif layer['layer_type'] == 'fc':
                if in_features is None:
                    # Transition from Conv to FC layers
                    self.conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    self.fc_layers.append(nn.Flatten())
                    in_features = input_channels

                out_features = layer['fc_layer_size']
                self.fc_layers.append(nn.Linear(in_features, out_features))
                self.fc_layers.append(nn.ReLU())
                in_features = out_features

        if in_features is None:
            self.conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc_layers.append(nn.Flatten())
            in_features = input_channels

        self.fc_layers.append(nn.Linear(in_features, self.board_size ** 2))

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
