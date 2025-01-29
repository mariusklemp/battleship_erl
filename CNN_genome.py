from neat.activations import ActivationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from random import choice, random, randint


class CNNGenomeConfig(object):
    __params = [
        ConfigParameter("input_channels", int),
        ConfigParameter("input_size", int),
        ConfigParameter("output_size", int),
        ConfigParameter("kernel_sizes", str),
        ConfigParameter("strides", str),
        ConfigParameter("paddings", str),
        ConfigParameter("out_channels_min", int),
        ConfigParameter("out_channels_max", int),
        ConfigParameter("fc_layer_size_min", int),
        ConfigParameter("fc_layer_size_max", int),
        ConfigParameter("compatibility_disjoint_coefficient", float),
        ConfigParameter("compatibility_weight_coefficient", float),
        ConfigParameter("activation_default", str),
        ConfigParameter("activation_options", str),
        ConfigParameter("num_cnn_layer", int),
        ConfigParameter("num_fc_layer", int),
        ConfigParameter("layer_add_prob", float),
        ConfigParameter("layer_delete_prob", float),
        ConfigParameter("conv_params_mutate_prob", float),
        ConfigParameter("conv_output_mutate_prob", float),
        ConfigParameter("conv_activation_mutate_prob", float),
    ]

    def __init__(self, params):
        # Parse base NEAT parameters
        self.activation_defs = ActivationFunctionSet()

        # Parse CNN-specific parameters
        for param in self.__params:
            setattr(self, param.name, param.interpret(params))

        # Parse string values into lists
        self.kernel_sizes = eval(self.kernel_sizes)
        self.strides = eval(self.strides)
        self.paddings = eval(self.paddings)
        self.activation_options = self.activation_options.split(",")

        # Pre-validate convolution parameters for the input size
        self.valid_conv_params = self.pre_validate_conv_params(self.input_size)

        # Define input and output keys
        self.input_keys = [
            -i - 1 for i in range(self.input_channels * self.input_size**2)
        ]
        self.output_keys = [i for i in range(self.output_size)]

    def pre_validate_conv_params(self, input_size):
        """
        Pre-validate and store combinations of kernel_size, stride, and padding
        that result in valid output sizes for the given input size.
        """
        valid_params = []
        for kernel_size in self.kernel_sizes:
            for stride in self.strides:
                for padding in self.paddings:
                    output_size = (
                        (input_size + 2 * padding - kernel_size) // stride
                    ) + 1
                    if output_size > 0:
                        valid_params.append((kernel_size, stride, padding))
        return valid_params

    def save(self, f):
        write_pretty_params(f, self, self.__params)


class CNNGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        return CNNGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key
        self.layer_config = []  # List of layer definitions
        self.fitness = None

    def configure_new(self, config):
        """Initialize a new genome with valid CNN layer configurations."""
        self.layer_config = []
        num_conv_layers = randint(
            1, config.num_cnn_layer
        )  # Decide the number of Conv layers
        num_fc_layers = randint(
            1, config.num_fc_layer
        )  # Decide the number of FC layers

        # Start with the input channels for the first layer
        in_channels = config.input_channels
        current_size = config.input_size  # Track the current spatial size of the input

        # Add Conv layers using pre-validated parameters
        for i in range(num_conv_layers):
            try:
                # Add Conv layer with input size and track the updated size
                new_layer = self.add_random_conv_layer(
                    config, in_channels, current_size
                )

                # Update in_channels and spatial size for the next layer
                new_current_size = (
                    (current_size + 2 * new_layer["padding"] - new_layer["kernel_size"])
                    // new_layer["stride"]
                ) + 1

                # Validate spatial size after each layer
                if new_current_size <= 0:
                    raise ValueError(f"Invalid spatial size after Conv Layer {i + 1}.")

                # Append the layer configuration with the updated input size
                self.layer_config.append(new_layer)
                in_channels = new_layer["out_channels"]
                current_size = (
                    new_current_size  # Update the spatial size for the next layer
                )

            except ValueError as e:
                print(f"[ERROR] {e}. Skipping Conv Layer {i + 1}.")
                continue

        # Ensure we have valid Conv layers
        if not self.layer_config or current_size <= 0:
            raise ValueError(
                "Failed to configure valid Conv layers. Aborting genome creation."
            )

        # Calculate the flattened size for the first FC layer
        flattened_size = max(1, current_size**2 * self.layer_config[-1]["out_channels"])

        # Add FC layers
        previous_fc_size = flattened_size  # Initialize with the flattened size
        for i in range(num_fc_layers):
            try:
                self.add_random_fc_layer(config, input_size=previous_fc_size)
                # Update the size for subsequent FC layers
                previous_fc_size = self.layer_config[-1]["fc_layer_size"]
            except ValueError as e:
                print(f"[ERROR] {e}. Skipping FC Layer {i + 1}.")

    def add_random_conv_layer(self, config, in_channels, input_size):
        """Add a Conv layer that fits with the input size from the previous layer."""
        # Ensure valid convolution parameters for the given input size
        valid_layers = []

        for kernel_size, stride, padding in config.valid_conv_params:
            # Calculate the output size for each valid parameter combination
            output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
            # Only add valid parameters that result in a positive output size
            if output_size > 0:
                valid_layers.append((kernel_size, stride, padding))

        if not valid_layers:
            raise ValueError(
                f"No valid Conv parameters found for input size {input_size}."
            )

        # Select a random valid convolution parameter combination
        kernel_size, stride, padding = choice(valid_layers)

        # Randomly select the number of output channels within the specified range
        out_channels = randint(config.out_channels_min, config.out_channels_max)

        # Store the layer configuration
        layer_config = {
            "layer_type": "conv",
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "out_channels": out_channels,
            "activation": config.activation_default,
            "in_channels": in_channels,
            "input_size": input_size,  # Store the input size for the current layer
        }

        return layer_config

    def add_random_fc_layer(self, config, input_size=None):
        """Add a new FC layer with random parameters."""
        if input_size is None:
            raise ValueError("Input size for the first FC layer cannot be None.")
        self.layer_config.append(
            {
                "layer_type": "fc",
                "fc_layer_size": randint(
                    config.fc_layer_size_min, config.fc_layer_size_max
                ),
                "activation": config.activation_default,
                "input_size": input_size,  # Dynamically determined for the first FC layer
            }
        )

    def calculate_conv_output_size(self, input_size, layer):
        """
        Calculate the output size of a Conv layer manually using its parameters.
        Ensures the size remains valid (positive and realistic).
        """
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        padding = layer["padding"]

        # Ensure kernel size is not larger than the input size
        if kernel_size > input_size:
            raise ValueError(
                f"Kernel size={kernel_size} is too large for input size={input_size}. "
                f"Reduce kernel size or increase input size."
            )

        # Calculate the new height/width of the tensor
        output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1

        # Validate output size
        if output_size <= 0:
            raise ValueError(
                f"Invalid configuration for Conv layer. "
                f"Kernel size={kernel_size}, Stride={stride}, Padding={padding}, "
                f"Input size={input_size}, Calculated output size={output_size}"
            )

        return output_size

    def mutate(self, config):
        """Mutate the genome by modifying or adding/removing layers."""
        # Add a new Conv layer
        if random() < config.layer_add_prob:
            if (
                len(self.layer_config) == 0
                or self.layer_config[-1]["layer_type"] == "conv"
            ):
                current_input_size = config.input_size
                in_channels = config.input_channels
                for layer in self.layer_config:
                    if layer["layer_type"] == "conv":
                        current_input_size = self.calculate_conv_output_size(
                            current_input_size, layer
                        )
                        in_channels = layer["out_channels"]

                new_layer = self.add_random_conv_layer(
                    config, in_channels, current_input_size
                )
                self.layer_config.append(new_layer)

        # Add a new FC layer only if Conv layers exist
        if random() < config.layer_add_prob and any(
            layer["layer_type"] == "conv" for layer in self.layer_config
        ):
            last_conv_layer = next(
                layer
                for layer in reversed(self.layer_config)
                if layer["layer_type"] == "conv"
            )
            current_input_size = config.input_size
            for layer in self.layer_config:
                if layer["layer_type"] == "conv":
                    current_input_size = self.calculate_conv_output_size(
                        current_input_size, layer
                    )

            input_size = max(1, current_input_size**2 * last_conv_layer["out_channels"])
            self.add_random_fc_layer(config, input_size=input_size)

        # Remove a layer if there are more than one
        if random() < config.layer_delete_prob and len(self.layer_config) > 1:
            # Ensure there is at least one conv and one fc layer
            conv_count = sum(
                1 for layer in self.layer_config if layer["layer_type"] == "conv"
            )
            fc_count = sum(
                1 for layer in self.layer_config if layer["layer_type"] == "fc"
            )

            # Check layer type before removing
            layer_to_remove = self.layer_config[
                -1
            ]  # Assume the last layer is targeted for removal
            if (layer_to_remove["layer_type"] == "conv" and conv_count > 1) or (
                layer_to_remove["layer_type"] == "fc" and fc_count > 1
            ):
                self.layer_config.pop()

        # Mutate existing layers
        for i, layer in enumerate(self.layer_config):
            if layer["layer_type"] == "conv":
                self._mutate_conv_layer(config, layer, i)
            elif layer["layer_type"] == "fc":
                self._mutate_fc_layer(config, layer, i)

        # Adjust input sizes dynamically after mutations
        self._adjust_layer_sizes(config)

    def _mutate_conv_layer(self, config, layer, index):
        """Mutate a Conv layer."""

        # Store the original layer configuration
        original_layer = layer.copy()

        # Mutate the convolution parameters (kernel size, stride, padding)
        if random() < config.conv_params_mutate_prob:
            # Filter valid conv parameters based on the input size

            # Get the input size for the next layer (if it exists)
            input_size_next_layer = (
                self.layer_config[index + 1]["input_size"]
                if index + 1 < len(self.layer_config)
                else config.output_size
            )

            # Filter valid conv parameters based on the input size for the current layer and the next layer
            valid_conv_params = [
                (kernel_size, stride, padding)
                for kernel_size, stride, padding in config.valid_conv_params
                if kernel_size
                <= input_size_next_layer  # Ensure the output size is compatible with the next layer
                and self._is_valid_conv_param(
                    layer["input_size"], kernel_size, stride, padding
                )
            ]

            if valid_conv_params:
                kernel_size, stride, padding = choice(valid_conv_params)
                layer["kernel_size"] = kernel_size
                layer["stride"] = stride
                layer["padding"] = padding
            else:
                print(
                    f"[ERROR] No valid convolution parameters found for Conv layer {index + 1}."
                )

        # Mutate the number of output channels
        if random() < config.conv_output_mutate_prob:
            new_out_channels = randint(config.out_channels_min, config.out_channels_max)
            if new_out_channels != layer["out_channels"]:
                layer["out_channels"] = new_out_channels

        # Randomly mutate the activation function
        if random() < config.conv_activation_mutate_prob:
            new_activation = choice(config.activation_options)
            if new_activation != layer["activation"]:
                layer["activation"] = new_activation

        # After mutation, ensure correct in_channels for the next layer
        if (
            index + 1 < len(self.layer_config)
            and self.layer_config[index + 1]["layer_type"] == "conv"
        ):
            self.layer_config[index + 1]["in_channels"] = layer["out_channels"]

        # Compare the original layer with the mutated one and print the differences
        changes = []
        for key in layer:
            if layer[key] != original_layer.get(key):
                changes.append(f"{key}: {original_layer.get(key)} -> {layer[key]}")

        if changes:
            # print(f"[DEBUG] Mutated Conv layer {index + 1}: {', '.join(changes)}")
            pass
        else:
            # print(f"[DEBUG] Conv layer {index + 1} was not mutated.")
            pass

    def _mutate_fc_layer(self, config, layer, index):
        """Mutate an FC layer."""
        original_layer = layer.copy()

        # Mutate FC layer size
        if random() < 0.3:
            new_fc_layer_size = randint(
                config.fc_layer_size_min, config.fc_layer_size_max
            )
            if new_fc_layer_size != layer["fc_layer_size"]:
                layer["fc_layer_size"] = new_fc_layer_size

        # Mutate activation function
        if random() < 0.3:
            new_activation = choice(config.activation_options)
            if new_activation != layer["activation"]:
                layer["activation"] = new_activation

        # Compare the original layer with the mutated one and print the differences
        changes = []
        for key in layer:
            if layer[key] != original_layer.get(key):
                changes.append(f"{key}: {original_layer.get(key)} -> {layer[key]}")

        if changes:
            # print(f"[DEBUG] Mutated FC layer {index + 1}: {', '.join(changes)}")
            pass

    def _is_valid_conv_param(self, input_size, kernel_size, stride, padding):
        """Check if a conv layer parameter combination is valid for a given input size."""
        output_size = (input_size - kernel_size + 2 * padding) // stride + 1
        return output_size > 0

    def _adjust_layer_sizes(self, config):
        """Adjust the sizes of layers (conv and fc) after mutations."""
        current_input_size = config.input_size
        flattened_size = None  # Track the flattened size for the first FC layer

        for i, layer in enumerate(self.layer_config):
            if layer["layer_type"] == "conv":
                # Check that the kernel size is valid for the input size
                if layer["kernel_size"] > current_input_size:
                    layer["kernel_size"] = (
                        current_input_size  # Adjust kernel size to fit input size
                    )

                # Calculate output size for Conv layers and update flattened size
                current_input_size = self.calculate_conv_output_size(
                    current_input_size, layer
                )

                # Validate output size before moving forward
                if current_input_size <= 0:
                    layer["kernel_size"] = max(
                        1, layer["kernel_size"] - 1
                    )  # Try reducing the kernel size
                    current_input_size = self.calculate_conv_output_size(
                        current_input_size, layer
                    )  # Recalculate with adjusted kernel size

                in_channels = layer["out_channels"]
                flattened_size = current_input_size**2 * in_channels

            elif layer["layer_type"] == "fc":
                # Adjust the input size for FC layers based on the last Conv layer's output size
                layer["input_size"] = flattened_size
                flattened_size = layer[
                    "fc_layer_size"
                ]  # Update for subsequent FC layers

    def configure_crossover(self, genome1, genome2, config):
        """
        Create a child genome by combining two parent genomes.
        Ensures Conv layers come before FC layers and maintains the same number of layers as the parents.
        """
        # print(f"[DEBUG] Starting crossover for genomes {genome1.key} and {genome2.key}.")
        self.layer_config = []

        # Separate Conv and FC layers from parents
        conv_layers_1 = [
            layer for layer in genome1.layer_config if layer["layer_type"] == "conv"
        ]
        conv_layers_2 = [
            layer for layer in genome2.layer_config if layer["layer_type"] == "conv"
        ]
        fc_layers_1 = [
            layer for layer in genome1.layer_config if layer["layer_type"] == "fc"
        ]
        fc_layers_2 = [
            layer for layer in genome2.layer_config if layer["layer_type"] == "fc"
        ]

        # Crossover Conv layers
        in_channels = config.input_channels
        current_input_size = config.input_size
        for i, (layer1, layer2) in enumerate(zip(conv_layers_1, conv_layers_2)):
            selected_layer = layer1 if random() < 0.5 else layer2
            selected_layer = (
                selected_layer.copy()
            )  # Avoid modifying the parent layers directly
            selected_layer["in_channels"] = in_channels  # Set the correct in_channels

            # Adjust kernel size dynamically if it's invalid
            while selected_layer["kernel_size"] > current_input_size:
                selected_layer["kernel_size"] -= 1

            # Validate the layer and calculate output size
            try:
                current_input_size = self.calculate_conv_output_size(
                    current_input_size, selected_layer
                )
                in_channels = selected_layer[
                    "out_channels"
                ]  # Update in_channels for the next layer
                self.layer_config.append(selected_layer)
            except ValueError as e:
                print(f"[ERROR] Could not fix Conv layer during crossover: {e}")

        # Calculate the flattened size for FC layers
        flattened_size = (
            max(1, current_input_size**2 * in_channels) if self.layer_config else None
        )

        # Crossover FC layers
        for i, (layer1, layer2) in enumerate(zip(fc_layers_1, fc_layers_2)):
            input_size = flattened_size if i == 0 else None
            selected_layer = layer1 if random() < 0.5 else layer2
            selected_layer = (
                selected_layer.copy()
            )  # Avoid modifying the parent layers directly
            if input_size is not None:
                selected_layer["input_size"] = input_size
            self.layer_config.append(selected_layer)

    def distance(self, other, config):
        distance = 0

        # Compare number of layers
        layer_diff = abs(len(self.layer_config) - len(other.layer_config))
        distance += layer_diff * config.compatibility_disjoint_coefficient

        # Compare layer types (conv vs. fc layers)
        min_layers = min(len(self.layer_config), len(other.layer_config))
        for i in range(min_layers):
            layer1 = self.layer_config[i]
            layer2 = other.layer_config[i]

            # Penalty for layer type mismatch
            if layer1["layer_type"] != layer2["layer_type"]:
                distance += 1

            # Compare convolutional layer parameters
            if layer1["layer_type"] == "conv" and layer2["layer_type"] == "conv":
                distance += abs(layer1["kernel_size"] - layer2["kernel_size"])
                distance += abs(layer1["stride"] - layer2["stride"])
                distance += abs(layer1["padding"] - layer2["padding"])
                distance += abs(layer1["out_channels"] - layer2["out_channels"])

            # Compare fully connected layer parameters
            if layer1["layer_type"] == "fc" and layer2["layer_type"] == "fc":
                distance += abs(layer1["input_size"] - layer2["input_size"])
                distance += abs(layer1["fc_layer_size"] - layer2["fc_layer_size"])

        # Add penalties for extra layers in longer genomes
        longer_genome = (
            self if len(self.layer_config) > len(other.layer_config) else other
        )
        for i in range(min_layers, len(longer_genome.layer_config)):
            distance += 1  # Each extra layer adds a penalty

        # Normalize distance (optional)
        # You may adjust this normalization to better fit your architecture and needs
        # The max_distance can be the maximum possible difference in the parameters
        max_distance = 1000  # This is a rough estimate, you may adjust this value.
        normalized_distance = distance / max_distance

        return normalized_distance

    def size(self):
        """
        Calculate the size of the genome.
        For a CNN genome, this can be the total number of trainable parameters.
        """
        total_params = 0
        for layer in self.layer_config:
            if layer["layer_type"] == "conv":
                # Conv2D parameters = (kernel_size^2 * in_channels * out_channels) + out_channels (bias)
                kernel_size = layer["kernel_size"]
                in_channels = layer.get(
                    "in_channels", 1
                )  # You might need to store this
                out_channels = layer["out_channels"]
                params = (kernel_size**2) * in_channels * out_channels + out_channels
                total_params += params
            elif layer["layer_type"] == "fc":
                # Fully connected parameters = (input_size * output_size) + output_size (bias)
                input_size = layer["input_size"]
                output_size = layer["fc_layer_size"]
                params = input_size * output_size + output_size
                total_params += params
        return total_params
