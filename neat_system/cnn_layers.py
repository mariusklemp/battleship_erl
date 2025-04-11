from random import random, choice, randint

from neat.attributes import FloatAttribute, StringAttribute, BoolAttribute
from neat.genes import BaseGene
import numpy as np

from neat_system.helpers import _mutate_weights_conv, _mutate_weights_fc, connection_distance

global_innovation_registry = {}
global_innovation_number = 0


def get_innovation_number(gene_type, signature):
    """
    Check if a gene with the same type and signature exists.
    If so, return its innovation number; otherwise, assign a new one.
    """
    global global_innovation_number
    key = (gene_type, signature)
    if key in global_innovation_registry:
        return global_innovation_registry[key]
    else:
        new_key = global_innovation_number
        global_innovation_registry[key] = new_key
        global_innovation_number += 1
        return new_key


class CNNConvGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('in_channels'),
        FloatAttribute('out_channels'),
        FloatAttribute('input_size'),
        FloatAttribute('kernel_size'),
        FloatAttribute('stride'),
        FloatAttribute('padding'),
        StringAttribute('activation'),
        FloatAttribute('weights'),
        FloatAttribute('biases'),
        BoolAttribute('enabled'),
    ]

    @classmethod
    def create(cls, config, in_channels, input_size):
        valid_layers = []
        for kernel_size, stride, padding in config.valid_conv_params:
            output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
            if output_size > 0:
                valid_layers.append((kernel_size, stride, padding))
        if not valid_layers:
            raise ValueError("No valid convolution parameters for input_size {}".format(input_size))
        kernel_size, stride, padding = choice(valid_layers)
        out_channels = randint(config.out_channels_min, config.out_channels_max)
        # Create a signature using only structural parameters.
        signature = (kernel_size, stride, padding, config.activation_function)
        key = get_innovation_number("CNNConvGene", signature)
        gene = cls(key)
        gene.kernel_size = kernel_size
        gene.stride = stride
        gene.padding = padding
        gene.out_channels = out_channels
        gene.activation = config.activation_function
        gene.in_channels = in_channels
        gene.input_size = input_size
        gene.enabled = True
        return gene

    def initialize_weights(self, config):
        k = int(self.kernel_size)
        in_c = int(self.in_channels)
        out_c = int(self.out_channels)
        self.weights = np.random.randn(out_c, in_c, k, k) * config.weight_init_stdev + config.weight_init_mean
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)
        self.biases = np.random.randn(out_c) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def mutate(self, config, mutate_weights=False):
        if random() < config.conv_params_mutate_prob:
            new_out_channels = randint(config.out_channels_min, config.out_channels_max)
            self.out_channels = new_out_channels

            weights, biases = _mutate_weights_conv(self,
                                                   config.weight_mutate_rate,
                                                   config.weight_mutate_power,
                                                   config.weight_replace_rate,
                                                   config.weight_min_value,
                                                   config.weight_max_value,
                                                   config)
            self.weights = weights
            self.biases = biases

        return self

    def distance(self, other, config):
        # Normalize differences by the maximum value to get a relative difference (avoid division by zero)
        kernel_diff = abs(self.kernel_size - other.kernel_size) / max(self.kernel_size, other.kernel_size)
        stride_diff = abs(self.stride - other.stride) / max(self.stride, other.stride)
        # For padding, if both are zero then use 0, otherwise normalize.
        pad_max = max(self.padding, other.padding) if max(self.padding, other.padding) != 0 else 1
        padding_diff = abs(self.padding - other.padding) / pad_max
        out_channels_diff = abs(self.out_channels - other.out_channels) / max(self.out_channels, other.out_channels)
        activation_diff = 0 if self.activation == other.activation else 1

        weight_diff = connection_distance(self, other)

        return (kernel_diff +
                stride_diff +
                padding_diff +
                out_channels_diff +
                activation_diff +
                weight_diff)

    def __str__(self):
        return (f"ConvGene(innov={self.key}, kernel={self.kernel_size}, stride={self.stride}, "
                f"pad={self.padding}, in_size={self.input_size}, in_ch={self.in_channels}, "
                f"out_ch={self.out_channels}, act={self.activation}, enabled={self.enabled})")

    def __repr__(self):
        return self.__str__()


class CNNPoolGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('input_size'),
        FloatAttribute('in_channels'),
        FloatAttribute('pool_size'),
        FloatAttribute('stride'),
        StringAttribute('pool_type'),
        BoolAttribute('enabled'),
    ]

    @classmethod
    def create(cls, config, in_channels, input_size):
        valid_params = []
        for pool_size in config.pool_sizes:
            if pool_size > input_size:
                continue
            for stride in config.pool_strides:
                output_size = ((input_size - pool_size) // stride) + 1
                if output_size > 0:
                    valid_params.append((pool_size, stride))
        if not valid_params:
            valid_params = [(1, 1)]
        pool_size, stride = choice(valid_params)
        pool_type = choice(config.pool_type)
        signature = (pool_size, stride, pool_type)
        key = get_innovation_number("CNNPoolGene", signature)
        gene = cls(key)
        gene.pool_size = pool_size
        gene.stride = stride
        gene.pool_type = pool_type
        gene.input_size = input_size
        gene.in_channels = in_channels
        gene.enabled = True
        return gene

    def mutate(self, config):
        return self

    def distance(self, other, config):
        pool_size_diff = abs(self.pool_size - other.pool_size) / max(self.pool_size, other.pool_size)
        stride_diff = abs(self.stride - other.stride) / max(self.stride, other.stride)
        pool_type_diff = 0 if self.pool_type == other.pool_type else 1

        return (pool_size_diff +
                stride_diff +
                pool_type_diff)

    def __str__(self):
        return (f"PoolGene(innov={self.key}, pool={self.pool_size}, stride={self.stride}, "
                f"type={self.pool_type}, in_ch={self.in_channels}, enabled={self.enabled})")

    def __repr__(self):
        return self.__str__()


class CNNFCGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('fc_layer_size'),
        StringAttribute('activation'),
        FloatAttribute('input_size'),
        FloatAttribute('weights'),
        FloatAttribute('biases'),
        BoolAttribute('enabled'),
    ]

    @classmethod
    def create(cls, config, input_size):
        fc_layer_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
        # Determine size category based on thresholds defined in your config.
        if fc_layer_size < 100:
            size_category = "small"
        elif fc_layer_size < 180:
            size_category = "medium"
        else:
            size_category = "large"
        # Group FC genes by size category and activation.
        signature = (size_category, config.activation_function)
        key = get_innovation_number("CNNFCGene", signature)
        gene = cls(key)
        gene.fc_layer_size = fc_layer_size
        gene.activation = config.activation_function
        gene.input_size = input_size
        gene.enabled = True
        return gene

    def initialize_weights(self, config):
        in_size = int(self.input_size)
        fc_size = int(self.fc_layer_size)
        self.weights = np.random.randn(fc_size, in_size) * config.weight_init_stdev + config.weight_init_mean
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)
        self.biases = np.random.randn(fc_size) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def mutate(self, config):
        if random() < config.fc_params_mutate_prob:
            # Based on the size category, mutate the FC layer size.
            if self.fc_layer_size < 100:
                new_fc_size = randint(config.fc_layer_size_min, 100)
            elif self.fc_layer_size < 180:
                new_fc_size = randint(100, 180)
            else:
                new_fc_size = randint(180, config.fc_layer_size_max)

            self.fc_layer_size = new_fc_size

            weights, biases = _mutate_weights_fc(self,
                                                 config.weight_mutate_rate,
                                                 config.weight_mutate_power,
                                                 config.weight_replace_rate,
                                                 config.weight_min_value,
                                                 config.weight_max_value,
                                                 config)
            self.weights = weights
            self.biases = biases

        return self

    def distance(self, other, config):
        # Use relative difference: if fc_layer_size is similar, this term will be small.
        fc_size_diff = abs(self.fc_layer_size - other.fc_layer_size) / max(self.fc_layer_size, other.fc_layer_size)
        activation_diff = 0 if self.activation == other.activation else 1
        weight_diff = connection_distance(self, other)

        return (fc_size_diff +
                activation_diff
                + weight_diff)

    def __str__(self):
        return (f"FCGene(innov={self.key}, fc_size={self.fc_layer_size}, input_size={self.input_size}, "
                f"activation={self.activation}, enabled={self.enabled}))")

    def __repr__(self):
        return self.__str__()
