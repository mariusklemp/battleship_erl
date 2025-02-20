import numpy as np
import copy
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


# ============================================================================
# Helper functions to adapt weights when dimensions change.
# ============================================================================

def adapt_conv_weights(old_weights, new_shape):
    """
    Adapt old convolution weights (old_shape: (old_out, old_in, old_k, old_k))
    to new_shape (new_out, new_in, new_k, new_k) by copying the overlapping
    region. For kernel dimensions, we center-crop (or pad) the kernels.
    """
    new_weights = np.random.randn(*new_shape) * 0.1  # new random init
    old_out, old_in, old_k, _ = old_weights.shape
    new_out, new_in, new_k, _ = new_shape

    # Overlap for channels:
    out_overlap = min(old_out, new_out)
    in_overlap = min(old_in, new_in)

    # For the kernel dimensions, we try to center the old kernel inside the new one.
    kernel_overlap = min(old_k, new_k)
    old_start = (old_k - kernel_overlap) // 2
    new_start = (new_k - kernel_overlap) // 2

    new_weights[:out_overlap, :in_overlap, new_start:new_start + kernel_overlap, new_start:new_start + kernel_overlap] = \
        old_weights[:out_overlap, :in_overlap, old_start:old_start + kernel_overlap,
        old_start:old_start + kernel_overlap]
    return new_weights


def adapt_fc_weights(old_weights, new_shape):
    """
    Adapt old fully-connected weights (old_shape: (old_out, old_in))
    to new_shape (new_out, new_in) by copying the overlapping region.
    """
    new_weights = np.random.randn(*new_shape) * 0.1
    old_out, old_in = old_weights.shape
    new_out, new_in = new_shape
    out_overlap = min(old_out, new_out)
    in_overlap = min(old_in, new_in)
    new_weights[:out_overlap, :in_overlap] = old_weights[:out_overlap, :in_overlap]
    return new_weights


def adapt_biases(old_biases, new_length):
    """
    Adapt biases: copy as many values as possible from old_biases into a new
    bias vector of length new_length.
    """
    new_biases = np.random.randn(new_length) * 0.1
    overlap = min(old_biases.shape[0], new_length)
    new_biases[:overlap] = old_biases[:overlap]
    return new_biases


# ============================================================================
# Gene definitions for CNN layers with weights
# ============================================================================

class CNNConvGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('kernel_size'),
        FloatAttribute('stride'),
        FloatAttribute('padding'),
        FloatAttribute('out_channels'),
        StringAttribute('activation'),
        FloatAttribute('in_channels'),
        FloatAttribute('input_size'),
        FloatAttribute('weights'),
        FloatAttribute('biases'),
    ]

    def initialize_weights(self, config):
        k = int(self.kernel_size)
        in_c = int(self.in_channels)
        out_c = int(self.out_channels)
        if config.weight_init_type in ["gaussian", "normal"]:
            self.weights = np.random.randn(out_c, in_c, k, k) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            self.weights = np.random.uniform(low_bound, high_bound, size=(out_c, in_c, k, k))
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)

        # Biases are initialized similarly:
        self.biases = np.random.randn(out_c) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def copy(self):
        return copy.deepcopy(self)

    def distance(self, other, config):
        d = abs(self.kernel_size - other.kernel_size)
        d += abs(self.stride - other.stride)
        d += abs(self.padding - other.padding)
        d += abs(self.out_channels - other.out_channels)
        if self.activation != other.activation:
            d += 1.0

        return d


class CNNFCGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('fc_layer_size'),
        StringAttribute('activation'),
        FloatAttribute('input_size'),
        FloatAttribute('weights'),
        FloatAttribute('biases'),
    ]

    def initialize_weights(self, config):
        in_size = int(self.input_size)
        fc_size = int(self.fc_layer_size)
        if config.weight_init_type in ["gaussian", "normal"]:
            self.weights = np.random.randn(fc_size, in_size) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            self.weights = np.random.uniform(low_bound, high_bound, size=(fc_size, in_size))
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)

        self.biases = np.random.randn(fc_size) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def copy(self):
        return copy.deepcopy(self)

    def distance(self, other, config):
        # Compute relative difference in fc_layer_size
        avg_fc = (self.fc_layer_size + other.fc_layer_size) / 2.0
        d_fc_size = abs(self.fc_layer_size - other.fc_layer_size) / (avg_fc if avg_fc != 0 else 1)

        # Compute relative difference in input_size
        avg_input = (self.input_size + other.input_size) / 2.0
        d_input = abs(self.input_size - other.input_size) / (avg_input if avg_input != 0 else 1)

        d = d_fc_size + d_input
        if self.activation != other.activation:
            d += 1.0  # Activation mismatch penalty.

        return d


# ============================================================================
# Genome configuration class (updated with weight mutation parameters)
# ============================================================================

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

        ConfigParameter("weight_init_mean", float),
        ConfigParameter("weight_init_stdev", float),
        ConfigParameter("weight_init_type", str),
        ConfigParameter("weight_max_value", float),
        ConfigParameter("weight_min_value", float),
        ConfigParameter("weight_mutate_power", float),
        ConfigParameter("weight_mutate_rate", float),
        ConfigParameter("weight_replace_rate", float),
        ConfigParameter("mutate_architecture", bool),
        ConfigParameter("mutate_weights", bool),
        ConfigParameter("crossover_architecture", bool),
    ]

    def __init__(self, params):
        self.activation_defs = ActivationFunctionSet()
        for param in self.__params:
            setattr(self, param.name, param.interpret(params))
        self.kernel_sizes = eval(self.kernel_sizes)
        self.strides = eval(self.strides)
        self.paddings = eval(self.paddings)
        self.activation_options = self.activation_options.split(",")
        self.valid_conv_params = self.pre_validate_conv_params(self.input_size)
        self.input_keys = [-i - 1 for i in range(self.input_channels * self.input_size ** 2)]
        self.output_keys = [i for i in range(self.output_size)]

    def pre_validate_conv_params(self, input_size):
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


# ============================================================================
# Genome class that uses the gene objects (updated to adjust weights on
# architecture changes using weight adaptation)
# ============================================================================

class CNNGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        return CNNGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key
        self.layer_config = []  # List of layer genes
        self.fitness = None

    def get_new_layer_key(self):
        return get_new_global_layer_key()

    def debug_print(self, label="Genome state", node=None):
        # Use the passed-in node, or default to self.
        target = node if node is not None else self
        print(f"--- {label} ---")
        for idx, gene in enumerate(target.layer_config):
            if isinstance(gene, CNNConvGene):
                print(f"Layer {idx}: CNNConvGene(key={gene.key}, kernel_size={gene.kernel_size}, "
                      f"stride={gene.stride}, padding={gene.padding}, out_channels={gene.out_channels}, "
                      f"in_channels={gene.in_channels}, input_size={gene.input_size}, activation={gene.activation})")
                print(f"         Weights shape: {gene.weights.shape}, Biases shape: {gene.biases.shape}")
                print(f"  Weights stats: mean={gene.weights.mean():.4f}, std={gene.weights.std():.4f}")
            elif isinstance(gene, CNNFCGene):
                print(f"Layer {idx}: CNNFCGene(key={gene.key}, fc_layer_size={gene.fc_layer_size}, "
                      f"input_size={gene.input_size}, activation={gene.activation})")
                print(f"         Weights shape: {gene.weights.shape}, Biases shape: {gene.biases.shape}")
                print(f"  Weights stats: mean={gene.weights.mean():.4f}, std={gene.weights.std():.4f}")
        print("----------")

    def configure_new(self, config):
        self.layer_config = []
        num_conv_layers = randint(1, config.num_cnn_layer)
        num_fc_layers = randint(1, config.num_fc_layer)
        in_channels = config.input_channels
        current_size = config.input_size

        # Create Conv layers.
        for i in range(num_conv_layers):
            try:
                new_gene = self.add_random_conv_layer(config, in_channels, current_size)
                new_gene.initialize_weights(config)
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

        # Create FC layers.
        flattened_size = max(1, current_size ** 2 * in_channels)
        previous_fc_size = flattened_size
        for i in range(num_fc_layers):
            try:
                fc_gene = self.add_random_fc_layer(config, input_size=previous_fc_size)
                fc_gene.initialize_weights(config)
                self.layer_config.append(fc_gene)
                previous_fc_size = fc_gene.fc_layer_size
            except ValueError as e:
                print(f"[ERROR] {e}. Skipping FC Layer {i + 1}.")
        self.enforce_valid_ordering()

    def add_random_conv_layer(self, config, in_channels, input_size):
        valid_layers = []
        for kernel_size, stride, padding in config.valid_conv_params:
            output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
            if output_size > 0:
                valid_layers.append((kernel_size, stride, padding))
        if not valid_layers:
            raise ValueError(f"No valid Conv parameters found for input size {input_size}.")
        kernel_size, stride, padding = choice(valid_layers)
        out_channels = randint(config.out_channels_min, config.out_channels_max)
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
        key = self.get_new_layer_key()
        gene = CNNFCGene(key)
        gene.fc_layer_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
        gene.activation = config.activation_default
        gene.input_size = input_size
        return gene

    def calculate_conv_output_size(self, input_size, conv_gene):
        kernel_size = conv_gene.kernel_size
        stride = conv_gene.stride
        padding = conv_gene.padding
        if kernel_size > input_size:
            raise ValueError(f"Kernel size={kernel_size} is too large for input size={input_size}.")
        output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
        if output_size <= 0:
            raise ValueError(f"Invalid configuration for Conv layer. Calculated output size={output_size}")
        return output_size

    def enforce_valid_ordering(self):
        conv_genes = [gene for gene in self.layer_config if isinstance(gene, CNNConvGene)]
        fc_genes = [gene for gene in self.layer_config if isinstance(gene, CNNFCGene)]
        self.layer_config = conv_genes + fc_genes

    def _mutate_weights_conv(self, conv_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                             weight_min_value, weight_max_value, config):
        # --- Noise-based mutation for weights ---
        mutation_mask = np.random.rand(*conv_gene.weights.shape) < weight_mutate_rate
        noise = np.random.randn(*conv_gene.weights.shape) * weight_mutate_power
        conv_gene.weights += mutation_mask * noise

        # --- Replacement mutation for weights ---
        replace_mask = np.random.rand(*conv_gene.weights.shape) < weight_replace_rate
        if config.weight_init_type in ["gaussian", "normal"]:
            new_values = np.random.randn(*conv_gene.weights.shape) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            new_values = np.random.uniform(low_bound, high_bound, size=conv_gene.weights.shape)
        conv_gene.weights = np.where(replace_mask, new_values, conv_gene.weights)

        # Clip weights to allowed range.
        conv_gene.weights = np.clip(conv_gene.weights, weight_min_value, weight_max_value)

        # --- Repeat the same process for biases ---
        mutation_mask_bias = np.random.rand(*conv_gene.biases.shape) < weight_mutate_rate
        noise_bias = np.random.randn(*conv_gene.biases.shape) * weight_mutate_power
        conv_gene.biases += mutation_mask_bias * noise_bias

        replace_mask_bias = np.random.rand(*conv_gene.biases.shape) < weight_replace_rate
        if config.weight_init_type in ["gaussian", "normal"]:
            new_biases = np.random.randn(*conv_gene.biases.shape) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            new_biases = np.random.uniform(low_bound, high_bound, size=conv_gene.biases.shape)
        conv_gene.biases = np.where(replace_mask_bias, new_biases, conv_gene.biases)
        conv_gene.biases = np.clip(conv_gene.biases, weight_min_value, weight_max_value)

    def _mutate_weights_fc(self, fc_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                           weight_min_value, weight_max_value, config):
        # Create a mask for weights to be mutated normally:
        mutation_mask = np.random.rand(*fc_gene.weights.shape) < weight_mutate_rate
        noise = np.random.randn(*fc_gene.weights.shape) * weight_mutate_power
        # Add noise to weights where mutation_mask is True:
        fc_gene.weights += mutation_mask * noise

        # Now, create another mask for weights to be replaced entirely:
        replace_mask = np.random.rand(*fc_gene.weights.shape) < weight_replace_rate
        if config.weight_init_type in ["gaussian", "normal"]:
            new_values = np.random.randn(*fc_gene.weights.shape) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            new_values = np.random.uniform(low_bound, high_bound, size=fc_gene.weights.shape)
        fc_gene.weights = np.where(replace_mask, new_values, fc_gene.weights)
        fc_gene.weights = np.clip(fc_gene.weights, weight_min_value, weight_max_value)

        # Do the same for biases:
        mutation_mask_bias = np.random.rand(*fc_gene.biases.shape) < weight_mutate_rate
        noise_bias = np.random.randn(*fc_gene.biases.shape) * weight_mutate_power
        fc_gene.biases += mutation_mask_bias * noise_bias

        replace_mask_bias = np.random.rand(*fc_gene.biases.shape) < weight_replace_rate
        if config.weight_init_type in ["gaussian", "normal"]:
            new_biases = np.random.randn(*fc_gene.biases.shape) * config.weight_init_stdev + config.weight_init_mean
        elif config.weight_init_type == "uniform":
            low_bound = max(config.weight_min_value, config.weight_init_mean - config.weight_init_stdev * 2)
            high_bound = min(config.weight_max_value, config.weight_init_mean + config.weight_init_stdev * 2)
            new_biases = np.random.uniform(low_bound, high_bound, size=fc_gene.biases.shape)
        fc_gene.biases = np.where(replace_mask_bias, new_biases, fc_gene.biases)
        fc_gene.biases = np.clip(fc_gene.biases, weight_min_value, weight_max_value)

    def mutate(self, config):
        # --- Architecture Mutation ---
        if config.mutate_architecture:
            # Possibly add a new Conv layer.
            if random() < config.layer_add_prob:
                current_input_size = config.input_size
                in_channels = config.input_channels
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                        in_channels = gene.out_channels
                new_gene = self.add_random_conv_layer(config, in_channels, current_input_size)
                new_gene.initialize_weights(config)
                fc_indices = [i for i, gene in enumerate(self.layer_config) if isinstance(gene, CNNFCGene)]
                if fc_indices:
                    self.layer_config.insert(fc_indices[0], new_gene)
                else:
                    self.layer_config.append(new_gene)

            # Possibly add a new FC layer.
            if random() < config.layer_add_prob and any(isinstance(g, CNNConvGene) for g in self.layer_config):
                conv_genes = [g for g in self.layer_config if isinstance(g, CNNConvGene)]
                last_conv_gene = conv_genes[-1]
                current_input_size = config.input_size
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                flattened_size = max(1, current_input_size ** 2 * last_conv_gene.out_channels)
                fc_gene = self.add_random_fc_layer(config, input_size=flattened_size)
                fc_gene.initialize_weights(config)
                self.layer_config.append(fc_gene)

            # Possibly remove a layer.
            if random() < config.layer_delete_prob and len(self.layer_config) > 1:
                fc_genes = [gene for gene in self.layer_config if isinstance(gene, CNNFCGene)]
                if len(fc_genes) > 1 or not isinstance(self.layer_config[-1], CNNFCGene):
                    self.layer_config.pop()

            # Mutate architecture parameters for each gene.
            for i, gene in enumerate(self.layer_config):
                if isinstance(gene, CNNConvGene):
                    self._mutate_conv_layer(config, gene, i)
                elif isinstance(gene, CNNFCGene):
                    self._mutate_fc_layer(config, gene, i)

        # --- Weight Mutation ---
        if config.mutate_weights:
            for i, gene in enumerate(self.layer_config):
                if isinstance(gene, CNNConvGene):
                    self._mutate_weights_conv(gene,
                                              config.weight_mutate_rate,
                                              config.weight_mutate_power,
                                              config.weight_replace_rate,
                                              config.weight_min_value,
                                              config.weight_max_value,
                                              config)
                elif isinstance(gene, CNNFCGene):
                    self._mutate_weights_fc(gene,
                                            config.weight_mutate_rate,
                                            config.weight_mutate_power,
                                            config.weight_replace_rate,
                                            config.weight_min_value,
                                            config.weight_max_value,
                                            config)

        self._adjust_layer_sizes(config)
        self.enforce_valid_ordering()

    def _mutate_conv_layer(self, config, conv_gene, index):
        architecture_changed = False
        if random() < config.conv_params_mutate_prob:
            valid_conv_params = [
                (kernel_size, stride, padding)
                for kernel_size, stride, padding in config.valid_conv_params
                if kernel_size <= conv_gene.input_size and
                   ((conv_gene.input_size + 2 * padding - kernel_size) // stride) + 1 > 0
            ]
            if valid_conv_params:
                kernel_size, stride, padding = choice(valid_conv_params)
                if (kernel_size != conv_gene.kernel_size or stride != conv_gene.stride or padding != conv_gene.padding):
                    conv_gene.kernel_size = kernel_size
                    conv_gene.stride = stride
                    conv_gene.padding = padding
                    architecture_changed = True
            else:
                print(f"[ERROR] No valid conv parameters found for mutation in layer {index + 1}.")

        if random() < config.conv_output_mutate_prob:
            new_out_channels = randint(config.out_channels_min, config.out_channels_max)
            if new_out_channels != conv_gene.out_channels:
                conv_gene.out_channels = new_out_channels
                architecture_changed = True

        if random() < config.activation_mutate_rate:
            conv_gene.activation = choice(config.activation_options)

        if architecture_changed:
            new_shape = (int(conv_gene.out_channels), int(conv_gene.in_channels),
                         int(conv_gene.kernel_size), int(conv_gene.kernel_size))
            conv_gene.weights = adapt_conv_weights(conv_gene.weights, new_shape)
            conv_gene.biases = adapt_biases(conv_gene.biases, new_shape[0])

        if index + 1 < len(self.layer_config):
            next_gene = self.layer_config[index + 1]
            if isinstance(next_gene, CNNConvGene):
                next_gene.in_channels = conv_gene.out_channels

    def _mutate_fc_layer(self, config, fc_gene, index):
        architecture_changed = False
        if random() < 0.3:
            new_fc_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
            if new_fc_size != fc_gene.fc_layer_size:
                fc_gene.fc_layer_size = new_fc_size
                architecture_changed = True
        if random() < 0.3:
            fc_gene.activation = choice(config.activation_options)
        if architecture_changed:
            new_shape = (int(fc_gene.fc_layer_size), int(fc_gene.input_size))
            fc_gene.weights = adapt_fc_weights(fc_gene.weights, new_shape)
            fc_gene.biases = adapt_biases(fc_gene.biases, new_shape[0])

    def _adjust_layer_sizes(self, config):
        current_input_size = config.input_size
        flattened_size = None

        for i, gene in enumerate(self.layer_config):
            if isinstance(gene, CNNConvGene):
                if i == 0:
                    gene.in_channels = config.input_channels
                else:
                    conv_genes_before = [g for g in self.layer_config[:i] if isinstance(g, CNNConvGene)]
                    if conv_genes_before:
                        prev_conv = conv_genes_before[-1]
                        gene.in_channels = prev_conv.out_channels

                if gene.kernel_size > current_input_size:
                    gene.kernel_size = current_input_size
                try:
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                except ValueError as e:
                    gene.kernel_size = max(1, gene.kernel_size - 1)
                    current_input_size = self.calculate_conv_output_size(current_input_size, gene)
                flattened_size = current_input_size ** 2 * gene.out_channels

                expected_shape = (int(gene.out_channels), int(gene.in_channels),
                                  int(gene.kernel_size), int(gene.kernel_size))
                if gene.weights.shape != expected_shape:
                    gene.weights = adapt_conv_weights(gene.weights, expected_shape)
                    gene.biases = adapt_biases(gene.biases, expected_shape[0])
            elif isinstance(gene, CNNFCGene):
                # Instead of checking gene.input_size, check the weight matrix shape.
                if gene.weights.shape[1] != flattened_size:
                    new_shape = (int(gene.fc_layer_size), int(flattened_size))
                    gene.weights = adapt_fc_weights(gene.weights, new_shape)
                    gene.biases = adapt_biases(gene.biases, new_shape[0])
                    gene.input_size = flattened_size
                flattened_size = gene.fc_layer_size

    def configure_crossover(self, genome1, genome2, config):
        if not config.crossover_architecture:
            # If architecture crossover is off, simply copy one parent's genome.
            chosen = genome1 if random() < 0.5 else genome2
            self.layer_config = copy.deepcopy(chosen.layer_config)
            self._adjust_layer_sizes(config)
            return

        # Otherwise, do the usual architecture crossover.
        self.layer_config = []
        conv_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNConvGene)]
        conv_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNConvGene)]
        fc_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNFCGene)]
        fc_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNFCGene)]

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

        flattened_size = max(1, current_input_size ** 2 * in_channels) if self.layer_config else None

        for i, (g1, g2) in enumerate(zip(fc_layers_1, fc_layers_2)):
            selected = g1.copy() if random() < 0.5 else g2.copy()
            if i == 0 and flattened_size is not None:
                selected.input_size = flattened_size
            self.layer_config.append(selected)

        self.enforce_valid_ordering()
        self._adjust_layer_sizes(config)

    def distance(self, other, config):
        # Build dictionaries keyed by gene key for fast lookup.
        self_genes = {gene.key: gene for gene in self.layer_config}
        other_genes = {gene.key: gene for gene in other.layer_config}
        all_keys = set(self_genes.keys()) | set(other_genes.keys())

        # --- Architecture (Layer) Distance ---
        arch_total = 0.0
        matching_arch_count = 0
        for key in all_keys:
            if key in self_genes and key in other_genes:
                arch_total += self_genes[key].distance(other_genes[key], config)
                matching_arch_count += 1

        disjoint_arch_total = 0.0
        for key in all_keys:
            if key not in self_genes:
                gene = other_genes[key]
                if isinstance(gene, CNNConvGene):
                    disjoint_arch_total += (abs(gene.kernel_size) +
                                            abs(gene.stride) +
                                            abs(gene.padding) +
                                            abs(gene.out_channels))
                elif isinstance(gene, CNNFCGene):
                    # Option 1: Use raw differences...
                    # disjoint_arch_total += (abs(gene.fc_layer_size) + abs(gene.input_size))
                    # Option 2: Use relative differences (normalize by typical values):
                    typical_fc = 100.0  # Example value; adjust based on your problem.
                    typical_in = 100.0
                    disjoint_arch_total += (abs(gene.fc_layer_size) / typical_fc +
                                            abs(gene.input_size) / typical_in)
            elif key not in other_genes:
                gene = self_genes[key]
                if isinstance(gene, CNNConvGene):
                    disjoint_arch_total += (abs(gene.kernel_size) +
                                            abs(gene.stride) +
                                            abs(gene.padding) +
                                            abs(gene.out_channels))
                elif isinstance(gene, CNNFCGene):
                    typical_fc = 100.0
                    typical_in = 100.0
                    disjoint_arch_total += (abs(gene.fc_layer_size) / typical_fc +
                                            abs(gene.input_size) / typical_in)
        max_nodes = max(len(self_genes), len(other_genes)) or 1
        arch_distance = (arch_total + disjoint_arch_total) / max_nodes

        # --- Connection (Weight & Bias) Distance ---
        conn_total = 0.0
        matching_conn_count = 0
        for key in all_keys:
            if key in self_genes and key in other_genes:
                gene1 = self_genes[key]
                gene2 = other_genes[key]
                if gene1.weights.shape == gene2.weights.shape:
                    d_weights = np.linalg.norm(gene1.weights - gene2.weights)
                else:
                    d_weights = 1.0
                if gene1.biases.shape == gene2.biases.shape:
                    d_biases = np.linalg.norm(gene1.biases - gene2.biases)
                else:
                    d_biases = 1.0
                conn_total += (d_weights + d_biases)
                matching_conn_count += 1

        disjoint_conn_total = 0.0
        for key in all_keys:
            if key not in self_genes:
                gene = other_genes[key]
                disjoint_conn_total += np.linalg.norm(gene.weights) + np.linalg.norm(gene.biases)
            elif key not in other_genes:
                gene = self_genes[key]
                disjoint_conn_total += np.linalg.norm(gene.weights) + np.linalg.norm(gene.biases)
        conn_distance = (conn_total + disjoint_conn_total) / max_nodes

        # --- Combined Distance ---
        # Introduce separate scaling parameters.
        arch_scale = getattr(config, 'arch_scale', 1.0)
        conn_scale = getattr(config, 'conn_scale', 1.0)
        total_distance = arch_scale * arch_distance + config.compatibility_weight_coefficient * conn_scale * conn_distance

        # Optionally, apply a global scaling factor.
        scale_factor = getattr(config, 'distance_scale', 100.0)
        scaled_distance = total_distance / scale_factor

        return scaled_distance

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
