import math
from random import random, choice, randint

from neat.attributes import FloatAttribute, StringAttribute, BoolAttribute
from neat.genes import BaseGene
import numpy as np

from neat_system.helpers import connection_distance

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
        signature = (
            kernel_size,
            stride,
            padding,
            out_channels,
            config.activation_function,
        )
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

        # He (Kaiming) std = sqrt(2 / fan_in), fan_in = in_c * k * k
        fan_in = in_c * k * k
        scale = math.sqrt(2.0 / fan_in)

        # draw from N(0, scale^2)
        self.weights = np.random.randn(out_c, in_c, k, k) * scale
        # biases → zero
        self.biases = np.zeros(out_c, dtype=np.float32)

    def mutate(self, config):
        """
        Mutate this conv gene’s weights and biases.
        - With probability config.conv_params_mutate_prob, perform mutation.
        - Each weight is either perturbed by Gaussian noise or replaced.
        - Same for biases.
        - Finally, clip to [weight_min_value, weight_max_value].
        """
        if random() < config.conv_params_mutate_prob:
            # 1) Mutate each weight in the 4D tensor
            for idx in np.ndindex(self.weights.shape):
                if random() < config.weight_mutate_rate:
                    # small Gaussian perturbation
                    delta = np.random.randn() * config.weight_mutate_power
                    self.weights[idx] += delta
                else:
                    # full replacement
                    self.weights[idx] = np.random.uniform(
                        config.weight_min_value,
                        config.weight_max_value
                    )

            # 2) Mutate each bias
            for i in range(self.biases.shape[0]):
                if random() < config.weight_mutate_rate:
                    delta = np.random.randn() * config.weight_mutate_power
                    self.biases[i] += delta
                else:
                    self.biases[i] = np.random.uniform(
                        config.weight_min_value,
                        config.weight_max_value
                    )

            # 3) Clip to allowable range
            np.clip(
                self.weights,
                config.weight_min_value,
                config.weight_max_value,
                out=self.weights
            )
            np.clip(
                self.biases,
                config.weight_min_value,
                config.weight_max_value,
                out=self.biases
            )

        return self

    def distance(self, other, config):
        """
        Returns (structural_diff, weight_diff) between this conv gene and other.
        structural_diff = sum of normalized kernel/stride/pad/out_channels/activation differences
        weight_diff     = connection_distance(self, other)
        """
        # 1) structural differences
        kernel_diff = abs(self.kernel_size - other.kernel_size) / max(self.kernel_size, other.kernel_size)
        stride_diff = abs(self.stride - other.stride) / max(self.stride, other.stride)
        pad_max = max(self.padding, other.padding) or 1
        padding_diff = abs(self.padding - other.padding) / pad_max
        out_ch_diff = abs(self.out_channels - other.out_channels) / max(self.out_channels, other.out_channels)
        activation_diff = 0 if self.activation == other.activation else 1

        struct_diff = (
                kernel_diff +
                stride_diff +
                padding_diff +
                out_ch_diff +
                activation_diff
        )

        # 2) weight difference
        weight_diff = connection_distance(self, other)

        return struct_diff, weight_diff

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
        """
        Returns (structural_diff, weight_diff) for pooling genes.
        weight_diff is always 0 here (no parameters).
        """
        pool_size_diff = abs(self.pool_size - other.pool_size) / max(self.pool_size, other.pool_size)
        stride_diff = abs(self.stride - other.stride) / max(self.stride, other.stride)
        type_diff = 0 if self.pool_type == other.pool_type else 1

        struct_diff = pool_size_diff + stride_diff + type_diff
        return struct_diff, 0.0

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
        # Group FC genes by size  and activation.
        signature = (
            fc_layer_size,
            config.activation_function,
        )
        key = get_innovation_number("CNNFCGene", signature)
        gene = cls(key)
        gene.fc_layer_size = fc_layer_size
        gene.activation = config.activation_function
        gene.input_size = input_size
        gene.enabled = True
        return gene

    def initialize_weights(self, config):
        in_f = int(self.input_size)
        out_f = int(self.fc_layer_size)

        # He std = sqrt(2 / fan_in), fan_in = in_f
        scale = math.sqrt(2.0 / in_f)

        self.weights = np.random.randn(out_f, in_f) * scale
        self.biases = np.zeros(out_f, dtype=np.float32)

    def mutate(self, config):
        # Decide whether to mutate this FC gene’s parameters at all
        if random() < config.fc_params_mutate_prob:
            # 1) Mutate weights
            for idx in np.ndindex(self.weights.shape):
                if random() < config.weight_mutate_rate:
                    # small Gaussian perturbation
                    delta = np.random.randn() * config.weight_mutate_power
                    self.weights[idx] += delta
                else:
                    # full replacement
                    self.weights[idx] = np.random.uniform(
                        config.weight_min_value,
                        config.weight_max_value
                    )
            # 2) Mutate biases
            for i in range(self.biases.shape[0]):
                if random() < config.weight_mutate_rate:
                    delta = np.random.randn() * config.weight_mutate_power
                    self.biases[i] += delta
                else:
                    self.biases[i] = np.random.uniform(
                        config.weight_min_value,
                        config.weight_max_value
                    )
            # 3) Clip everything to the allowable range
            np.clip(
                self.weights,
                config.weight_min_value,
                config.weight_max_value,
                out=self.weights
            )
            np.clip(
                self.biases,
                config.weight_min_value,
                config.weight_max_value,
                out=self.biases
            )

        return self

    def distance(self, other, config):
        """
        Returns (structural_diff, weight_diff) for FC genes.
        structural_diff = normalized fc_size + activation difference
        weight_diff     = connection_distance(self, other)
        """
        size_diff = abs(self.fc_layer_size - other.fc_layer_size) / max(self.fc_layer_size, other.fc_layer_size)
        activation_diff = 0 if self.activation == other.activation else 1

        struct_diff = size_diff + activation_diff
        weight_diff = connection_distance(self, other)
        return struct_diff, weight_diff

    def __str__(self):
        return (f"FCGene(innov={self.key}, fc_size={self.fc_layer_size}, input_size={self.input_size}, "
                f"activation={self.activation}, enabled={self.enabled}))")

    def __repr__(self):
        return self.__str__()
