from random import choice, random, randint
from neat.activations import ActivationFunctionSet
from neat.attributes import FloatAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene

global_innovation_number = 0


def get_new_global_layer_key():
    global global_innovation_number
    key = global_innovation_number
    global_innovation_number += 1
    return key


# -------------------------------------------------------------------
# Gene definitions for CNN layers
# -------------------------------------------------------------------

class CNNConvGene(BaseGene):
    """
    A gene representing a convolutional (conv) layer.
    Attributes:
      - kernel_size, stride, padding (floats; usually representing integers)
      - out_channels: number of output channels.
      - activation: name of the activation function.
      - in_channels: number of input channels.
      - input_size: spatial size of the input feature map.
    """
    _gene_attributes = [
        FloatAttribute('kernel_size'),
        FloatAttribute('stride'),
        FloatAttribute('padding'),
        FloatAttribute('out_channels'),
        StringAttribute('activation'),
        FloatAttribute('in_channels'),
        FloatAttribute('input_size'),
    ]

    def distance(self, other, config):
        d = abs(self.kernel_size - other.kernel_size)
        d += abs(self.stride - other.stride)
        d += abs(self.padding - other.padding)
        d += abs(self.out_channels - other.out_channels)
        if self.activation != other.activation:
            d += 1.0
        return d


class CNNFCGene(BaseGene):
    """
    A gene representing a fully-connected (fc) layer.
    Attributes:
      - fc_layer_size: the number of neurons in this layer.
      - activation: name of the activation function.
      - input_size: the size of the flattened input.
    """
    _gene_attributes = [
        FloatAttribute('fc_layer_size'),
        StringAttribute('activation'),
        FloatAttribute('input_size'),
    ]

    def distance(self, other, config):
        d = abs(self.fc_layer_size - other.fc_layer_size)
        if self.activation != other.activation:
            d += 1.0
        d += abs(self.input_size - other.input_size)
        return d


# -------------------------------------------------------------------
# Genome configuration class
# -------------------------------------------------------------------

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
        ConfigParameter("activation_mutate_rate", float),
        ConfigParameter("activation_options", str),
        ConfigParameter("num_cnn_layer", int),
        ConfigParameter("num_fc_layer", int),
        ConfigParameter("layer_add_prob", float),
        ConfigParameter("layer_delete_prob", float),
        ConfigParameter("conv_params_mutate_prob", float),
        ConfigParameter("conv_output_mutate_prob", float),
    ]

    def __init__(self, params):
        # Parse base NEAT parameters
        self.activation_defs = ActivationFunctionSet()

        # Parse CNN-specific parameters from the parameter dictionary.
        for param in self.__params:
            setattr(self, param.name, param.interpret(params))

        # Parse string values into lists
        self.kernel_sizes = eval(self.kernel_sizes)
        self.strides = eval(self.strides)
        self.paddings = eval(self.paddings)
        self.activation_options = self.activation_options.split(",")

        # Pre-validate convolution parameters for the input size
        self.valid_conv_params = self.pre_validate_conv_params(self.input_size)

        # Define input and output keys (if needed elsewhere)
        self.input_keys = [
            -i - 1 for i in range(self.input_channels * self.input_size ** 2)
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
                    output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
                    if output_size > 0:
                        valid_params.append((kernel_size, stride, padding))
        return valid_params

    def save(self, f):
        write_pretty_params(f, self, self.__params)


# -------------------------------------------------------------------
# Genome class that uses the gene objects
# -------------------------------------------------------------------

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

    def get_new_layer_key(self):
        return get_new_global_layer_key()

    def configure_new(self, config):
        """Initialize a new genome with valid CNN layer configurations."""
        self.layer_config = []
        num_conv_layers = randint(1, config.num_cnn_layer)  # Random number of Conv layers
        num_fc_layers = randint(1, config.num_fc_layer)  # Random number of FC layers

        # Start with the input channels and spatial size for the first layer
        in_channels = config.input_channels
        current_size = config.input_size

        # --- Create Conv layers ---
        for i in range(num_conv_layers):
            try:
                new_gene = self.add_random_conv_layer(config, in_channels, current_size)
                # Calculate the new spatial size from the Conv layer parameters
                new_current_size = ((current_size + 2 * new_gene.padding - new_gene.kernel_size) //
                                    new_gene.stride) + 1

                if new_current_size <= 0:
                    raise ValueError(f"Invalid spatial size after Conv Layer {i + 1}.")

                self.layer_config.append(new_gene)
                in_channels = new_gene.out_channels
                current_size = new_current_size

            except ValueError as e:
                print(f"[ERROR] {e}. Skipping Conv Layer {i + 1}.")
                continue

        if not self.layer_config or current_size <= 0:
            raise ValueError("Failed to configure valid Conv layers. Aborting genome creation.")

        # --- Create FC layers ---
        # Calculate flattened size for the first FC layer.
        flattened_size = max(1, current_size ** 2 * in_channels)
        previous_fc_size = flattened_size
        for i in range(num_fc_layers):
            try:
                fc_gene = self.add_random_fc_layer(config, input_size=previous_fc_size)
                self.layer_config.append(fc_gene)
                previous_fc_size = fc_gene.fc_layer_size
            except ValueError as e:
                print(f"[ERROR] {e}. Skipping FC Layer {i + 1}.")

        # Enforce valid ordering: conv genes first, then fc genes.
        self.enforce_valid_ordering()

    def add_random_conv_layer(self, config, in_channels, input_size):
        """Create and return a new CNNConvGene that is valid for the given input size."""
        valid_layers = []
        # Filter valid conv parameters based on the input size.
        for kernel_size, stride, padding in config.valid_conv_params:
            output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
            if output_size > 0:
                valid_layers.append((kernel_size, stride, padding))

        if not valid_layers:
            raise ValueError(f"No valid Conv parameters found for input size {input_size}.")

        # Select a random valid convolution parameter combination.
        kernel_size, stride, padding = choice(valid_layers)
        out_channels = randint(config.out_channels_min, config.out_channels_max)

        # Create a new conv gene.
        key = self.get_new_layer_key()
        gene = CNNConvGene(key)
        gene.kernel_size = kernel_size
        gene.stride = stride
        gene.padding = padding
        gene.out_channels = out_channels
        gene.activation = config.activation_default
        gene.in_channels = in_channels
        gene.input_size = input_size
        return gene

    def add_random_fc_layer(self, config, input_size):
        """Create and return a new CNNFCGene with random parameters."""
        key = self.get_new_layer_key()
        gene = CNNFCGene(key)
        gene.fc_layer_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
        gene.activation = config.activation_default
        gene.input_size = input_size
        return gene

    def calculate_conv_output_size(self, input_size, conv_gene):
        """
        Calculate the output spatial size of a Conv layer using its gene parameters.
        """
        kernel_size = conv_gene.kernel_size
        stride = conv_gene.stride
        padding = conv_gene.padding

        if kernel_size > input_size:
            raise ValueError(
                f"Kernel size={kernel_size} is too large for input size={input_size}. "
                f"Reduce kernel size or increase input size."
            )

        output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1

        if output_size <= 0:
            raise ValueError(
                f"Invalid configuration for Conv layer. "
                f"Kernel size={kernel_size}, Stride={stride}, Padding={padding}, "
                f"Input size={input_size}, Calculated output size={output_size}"
            )

        return output_size

    def enforce_valid_ordering(self):
        """
        Reorder the layer_config so that all convolutional genes come before
        fully connected genes.
        """
        conv_genes = [gene for gene in self.layer_config if isinstance(gene, CNNConvGene)]
        fc_genes = [gene for gene in self.layer_config if isinstance(gene, CNNFCGene)]
        self.layer_config = conv_genes + fc_genes

    def mutate(self, config):
        """Mutate the genome by modifying or adding/removing layers."""
        # --- Possibly add a new Conv layer ---
        if random() < config.layer_add_prob:
            # Determine current input size and in_channels from existing Conv layers.
            current_input_size = config.input_size
            in_channels = config.input_channels
            for gene in self.layer_config:
                if isinstance(gene, CNNConvGene):
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                    in_channels = gene.out_channels

            new_gene = self.add_random_conv_layer(config, in_channels, current_input_size)
            # Insert the new conv gene before the first FC gene.
            fc_indices = [i for i, gene in enumerate(self.layer_config) if isinstance(gene, CNNFCGene)]
            if fc_indices:
                self.layer_config.insert(fc_indices[0], new_gene)
            else:
                self.layer_config.append(new_gene)

        # --- Possibly add a new FC layer (only if at least one Conv exists) ---
        if random() < config.layer_add_prob and any(isinstance(g, CNNConvGene) for g in self.layer_config):
            # Find the last Conv gene.
            conv_genes = [g for g in self.layer_config if isinstance(g, CNNConvGene)]
            last_conv_gene = conv_genes[-1]
            # Recalculate spatial size from the Conv layers.
            current_input_size = config.input_size
            for gene in self.layer_config:
                if isinstance(gene, CNNConvGene):
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
            flattened_size = max(1, current_input_size ** 2 * last_conv_gene.out_channels)

            fc_gene = self.add_random_fc_layer(config, input_size=flattened_size)
            # Append FC gene at the end.
            self.layer_config.append(fc_gene)

        # --- Possibly remove a layer if more than one exists ---
        if random() < config.layer_delete_prob and len(self.layer_config) > 1:
            # Check if deletion would remove the only FC gene.
            fc_genes = [gene for gene in self.layer_config if isinstance(gene, CNNFCGene)]
            if len(fc_genes) > 1 or not isinstance(self.layer_config[-1], CNNFCGene):
                self.layer_config.pop()

        # --- Mutate existing layers ---
        for i, gene in enumerate(self.layer_config):
            if isinstance(gene, CNNConvGene):
                self._mutate_conv_layer(config, gene, i)
            elif isinstance(gene, CNNFCGene):
                self._mutate_fc_layer(config, gene, i)

        # Adjust input sizes after mutation.
        self._adjust_layer_sizes(config)
        # Finally, enforce the proper ordering.
        self.enforce_valid_ordering()

    def _mutate_conv_layer(self, config, conv_gene, index):
        """Mutate a convolutional gene."""
        if random() < config.conv_params_mutate_prob:
            valid_conv_params = [
                (kernel_size, stride, padding)
                for kernel_size, stride, padding in config.valid_conv_params
                if kernel_size <= conv_gene.input_size and
                   ((conv_gene.input_size + 2 * padding - kernel_size) // stride) + 1 > 0
            ]
            if valid_conv_params:
                kernel_size, stride, padding = choice(valid_conv_params)
                conv_gene.kernel_size = kernel_size
                conv_gene.stride = stride
                conv_gene.padding = padding
            else:
                print(f"[ERROR] No valid conv parameters found for mutation in layer {index + 1}.")

        if random() < config.conv_output_mutate_prob:
            new_out_channels = randint(config.out_channels_min, config.out_channels_max)
            conv_gene.out_channels = new_out_channels

        if random() < config.activation_mutate_rate:
            conv_gene.activation = choice(config.activation_options)

        if index + 1 < len(self.layer_config):
            next_gene = self.layer_config[index + 1]
            if isinstance(next_gene, CNNConvGene):
                next_gene.in_channels = conv_gene.out_channels

    def _mutate_fc_layer(self, config, fc_gene, index):
        """Mutate a fully-connected gene."""
        if random() < 0.3:
            fc_gene.fc_layer_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
        if random() < 0.3:
            fc_gene.activation = choice(config.activation_options)

    def _adjust_layer_sizes(self, config):
        """
        After mutations, adjust each layer’s input size based on the output of preceding Conv layers.
        """
        current_input_size = config.input_size
        flattened_size = None

        for i, gene in enumerate(self.layer_config):
            if isinstance(gene, CNNConvGene):
                if gene.kernel_size > current_input_size:
                    gene.kernel_size = current_input_size
                try:
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                except ValueError as e:
                    gene.kernel_size = max(1, gene.kernel_size - 1)
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                flattened_size = current_input_size ** 2 * gene.out_channels
            elif isinstance(gene, CNNFCGene):
                gene.input_size = flattened_size
                flattened_size = gene.fc_layer_size

    def configure_crossover(self, genome1, genome2, config):
        """
        Create a child genome by combining two parent genomes.
        Conv genes come first followed by FC genes.
        """
        self.layer_config = []

        # Separate layers by type.
        conv_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNConvGene)]
        conv_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNConvGene)]
        fc_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNFCGene)]
        fc_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNFCGene)]

        # --- Crossover for Conv layers ---
        in_channels = config.input_channels
        current_input_size = config.input_size
        for i, (g1, g2) in enumerate(zip(conv_layers_1, conv_layers_2)):
            selected = g1.copy() if random() < 0.5 else g2.copy()
            selected.in_channels = in_channels
            while selected.kernel_size > current_input_size and selected.kernel_size > 1:
                selected.kernel_size -= 1
            try:
                current_input_size = self.calculate_conv_output_size(current_input_size, selected)
                in_channels = selected.out_channels
                self.layer_config.append(selected)
            except ValueError as e:
                print(f"[ERROR] Crossover Conv layer adjustment failed: {e}")

        # --- Determine flattened size for FC layers ---
        flattened_size = max(1, current_input_size ** 2 * in_channels) if self.layer_config else None

        # --- Crossover for FC layers ---
        for i, (g1, g2) in enumerate(zip(fc_layers_1, fc_layers_2)):
            selected = g1.copy() if random() < 0.5 else g2.copy()
            if i == 0 and flattened_size is not None:
                selected.input_size = flattened_size
            self.layer_config.append(selected)

        # Finally, enforce the proper ordering.
        self.enforce_valid_ordering()

    def distance(self, other, config):
        """Compute a distance measure between two genomes."""
        distance = 0
        layer_diff = abs(len(self.layer_config) - len(other.layer_config))
        distance += layer_diff * config.compatibility_disjoint_coefficient
        min_layers = min(len(self.layer_config), len(other.layer_config))
        for i in range(min_layers):
            gene1 = self.layer_config[i]
            gene2 = other.layer_config[i]
            if type(gene1) != type(gene2):
                distance += 1
            if isinstance(gene1, CNNConvGene) and isinstance(gene2, CNNConvGene):
                distance += abs(gene1.kernel_size - gene2.kernel_size)
                distance += abs(gene1.stride - gene2.stride)
                distance += abs(gene1.padding - gene2.padding)
                distance += abs(gene1.out_channels - gene2.out_channels)
            elif isinstance(gene1, CNNFCGene) and isinstance(gene2, CNNFCGene):
                distance += abs(gene1.input_size - gene2.input_size)
                distance += abs(gene1.fc_layer_size - gene2.fc_layer_size)
        longer = self if len(self.layer_config) > len(other.layer_config) else other
        for i in range(min_layers, len(longer.layer_config)):
            distance += 1
        max_distance = 1000
        normalized_distance = distance / max_distance
        return normalized_distance

    def size(self):
        total_params = 0
        for gene in self.layer_config:
            if isinstance(gene, CNNConvGene):
                params = (gene.kernel_size ** 2) * gene.in_channels * gene.out_channels + gene.out_channels
                total_params += params
            elif isinstance(gene, CNNFCGene):
                params = gene.input_size * gene.fc_layer_size + gene.fc_layer_size
                total_params += params
        return total_params

    def __str__(self):
        s = f"Genome {self.key}:\n"
        for gene in self.layer_config:
            if isinstance(gene, CNNConvGene):
                s += (f"  Conv: kernel_size={gene.kernel_size}, stride={gene.stride}, padding={gene.padding}, "
                      f"in_channels={gene.in_channels}, out_channels={gene.out_channels}, "
                      f"input_size={gene.input_size}, activation={gene.activation}\n")
            elif isinstance(gene, CNNFCGene):
                s += (f"  FC: input_size={gene.input_size}, fc_layer_size={gene.fc_layer_size}, "
                      f"activation={gene.activation}\n")
        return s
