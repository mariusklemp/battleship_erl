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

    @classmethod
    def from_spec(cls, config, spec, in_channels, input_size):
        # build the same signature you’d get in `create()`
        signature = (
            spec['kernel_size'],
            spec['stride'],
            spec['padding'],
            spec['out_channels'],
            config.activation_function,
        )
        key = get_innovation_number("CNNConvGene", signature)
        gene = cls(key)
        gene.in_channels = in_channels
        gene.input_size = input_size
        gene.kernel_size = spec['kernel_size']
        gene.stride = spec['stride']
        gene.padding = spec['padding']
        gene.out_channels = spec['out_channels']
        gene.activation = config.activation_function
        gene.enabled = True
        gene.initialize_weights(config)
        return gene

    def initialize_weights(self, config):
        # For ConvGene:
        k = int(self.kernel_size)
        in_c = int(self.in_channels)
        out_c = int(self.out_channels)
        fan_in = in_c * k * k
        scale = math.sqrt(2.0 / fan_in)  # He‐init σ

        self.weights = np.random.randn(out_c, in_c, k, k) * scale
        self.biases = np.zeros(out_c, dtype=np.float32)

        # remember for mutation
        self._weight_init_std = scale

    def mutate(self, config):
        # 1) Gate on whether to touch this conv’s params at all
        if random() >= config.conv_params_mutate_prob:
            return self

        low, high = config.weight_min_value, config.weight_max_value
        rate_mut = config.weight_mutate_rate
        rate_rep = config.weight_replace_rate
        total_rate = rate_mut + rate_rep
        p_perturb = rate_mut / total_rate

        # 2) Compute layer‐specific Gaussian σ = init_std × frac
        layer_sigma = self._weight_init_std * config.weight_mutate_frac

        # --- WEIGHTS ---
        W = self.weights
        mask_change = np.random.rand(*W.shape) < total_rate
        mask_perturb = np.random.rand(*W.shape) < p_perturb

        # a) perturbation deltas, b) full‐range replacements
        deltas = np.random.randn(*W.shape) * layer_sigma
        repls = np.random.uniform(low, high, size=W.shape)

        # c) apply
        W[mask_change & mask_perturb] += deltas[mask_change & mask_perturb]
        W[mask_change & ~mask_perturb] = repls[mask_change & ~mask_perturb]

        # d) clip into bounds
        np.clip(W, low, high, out=W)

        # --- BIASES ---
        B = self.biases
        mask_change = np.random.rand(*B.shape) < total_rate
        mask_perturb = np.random.rand(*B.shape) < p_perturb

        bdeltas = np.random.randn(*B.shape) * layer_sigma
        brepls = np.random.uniform(low, high, size=B.shape)

        B[mask_change & mask_perturb] += bdeltas[mask_change & mask_perturb]
        B[mask_change & ~mask_perturb] = brepls[mask_change & ~mask_perturb]
        np.clip(B, low, high, out=B)

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

    @classmethod
    def from_spec(cls, config, spec, input_size):
        signature = (spec['out_features'], config.activation_function)
        key = get_innovation_number("CNNFCGene", signature)
        gene = cls(key)
        gene.fc_layer_size = spec['out_features']
        gene.activation = config.activation_function
        gene.input_size = spec['in_features']
        gene.dynamic = spec.get('dynamic', True)
        gene.last = spec.get('last', False)
        gene.enabled = True
        gene.initialize_weights(config)
        return gene

    def initialize_weights(self, config):
        in_f = int(self.input_size)
        out_f = int(self.fc_layer_size)
        fan_in = in_f
        scale = math.sqrt(2.0 / fan_in)

        self.weights = np.random.randn(out_f, in_f) * scale
        self.biases = np.zeros(out_f, dtype=np.float32)

        # ← and here as well
        self._weight_init_std = scale

    def mutate(self, config):
        # 1) Gate on whether this FC layer’s weights/biases should be touched
        if random() >= config.fc_params_mutate_prob:
            return self

        low, high = config.weight_min_value, config.weight_max_value
        rate_mut = config.weight_mutate_rate
        rate_rep = config.weight_replace_rate
        total_rate = rate_mut + rate_rep
        p_perturb = rate_mut / total_rate

        # Compute this layer’s Gaussian σ as a fraction of its He‐init std
        layer_sigma = self._weight_init_std * config.weight_mutate_frac

        # --- WEIGHTS ---
        W = self.weights
        mask_change = np.random.rand(*W.shape) < total_rate
        mask_perturb = np.random.rand(*W.shape) < p_perturb

        # a) Gaussian perturbations
        deltas = np.random.randn(*W.shape) * layer_sigma
        # b) Uniform replacements
        repls = np.random.uniform(low, high, size=W.shape)

        # c) Apply them
        W[mask_change & mask_perturb] += deltas[mask_change & mask_perturb]
        W[mask_change & ~mask_perturb] = repls[mask_change & ~mask_perturb]

        # d) Clip back into [low, high]
        np.clip(W, low, high, out=W)

        # --- BIASES ---
        B = self.biases
        mask_change = np.random.rand(*B.shape) < total_rate
        mask_perturb = np.random.rand(*B.shape) < p_perturb

        bdeltas = np.random.randn(*B.shape) * layer_sigma
        brepls = np.random.uniform(low, high, size=B.shape)

        B[mask_change & mask_perturb] += bdeltas[mask_change & mask_perturb]
        B[mask_change & ~mask_perturb] = brepls[mask_change & ~mask_perturb]
        np.clip(B, low, high, out=B)

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
