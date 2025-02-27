import copy
from random import random, choice, randint

from neat.attributes import FloatAttribute, StringAttribute
from neat.genes import BaseGene
import numpy as np

global_innovation_number = 0


def get_new_global_layer_key():
    global global_innovation_number
    key = global_innovation_number
    global_innovation_number += 1
    return key


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
        key = get_new_global_layer_key()
        gene = cls(key)
        gene.kernel_size = kernel_size
        gene.stride = stride
        gene.padding = padding
        gene.out_channels = out_channels
        gene.activation = config.activation_function
        gene.in_channels = in_channels
        gene.input_size = input_size
        return gene


    def initialize_weights(self, config):
        k = int(self.kernel_size)
        in_c = int(self.in_channels)
        out_c = int(self.out_channels)
        self.weights = np.random.randn(out_c, in_c, k, k) * config.weight_init_stdev + config.weight_init_mean
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)
        self.biases = np.random.randn(out_c) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def mutate(self, config):
        # Mutate convolution parameters if applicable.
        if random() < config.conv_params_mutate_prob:
            valid_conv_params = [
                (kernel_size, stride, padding)
                for kernel_size, stride, padding in config.valid_conv_params
                if kernel_size <= self.input_size and ((self.input_size + 2 * padding - kernel_size) // stride) + 1 > 0
            ]

            new_params = choice(valid_conv_params)
            if new_params != (self.kernel_size, self.stride, self.padding):
                self.kernel_size, self.stride, self.padding = new_params

        # Mutate out_channels.
        if random() < config.conv_output_mutate_prob:
            new_out_channels = randint(config.out_channels_min, config.out_channels_max)
            if new_out_channels != self.out_channels:
                self.out_channels = new_out_channels

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return (f"ConvGene(kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, activation={self.activation})")

    def __repr__(self):
        return self.__str__()


class CNNPoolGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('input_size'),
        FloatAttribute('in_channels'),
        FloatAttribute('pool_size'),
        FloatAttribute('stride'),
        StringAttribute('pool_type'),
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
        key = get_new_global_layer_key()
        gene = cls(key)
        gene.pool_size = pool_size
        gene.stride = stride
        gene.pool_type = pool_type
        gene.input_size = input_size
        gene.in_channels = in_channels
        return gene

    def mutate(self, config):
        valid_pool_params = []
        # Only consider pool sizes that are <= current input_size.
        for pool_size in config.pool_sizes:
            if pool_size > self.input_size:
                continue  # Skip pool sizes that are too large.
            for stride in config.pool_strides:
                output_size = ((self.input_size - pool_size) // stride) + 1
                if output_size > 0:
                    valid_pool_params.append((pool_size, stride))
        print(f"[DEBUG] PoolGene.mutate: input_size={self.input_size}, valid_params={valid_pool_params}")
        if valid_pool_params:
            new_params = choice(valid_pool_params)
            if new_params != (self.pool_size, self.stride):
                self.pool_size, self.stride = new_params
        else:
            # Fallback: use a no-op pooling.
            print(f"[ERROR] No valid pool parameters for input_size {self.input_size}; fallback to (1,1)")
            self.pool_size, self.stride = 1, 1

        # Optionally mutate pool type.
        if random() < 0.3:
            self.pool_type = choice(config.pool_type)

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return (f"PoolGene(pool_size={self.pool_size}, stride={self.stride}, "
                f"pool_type={self.pool_type}, in_channels={self.in_channels})")

    def __repr__(self):
        return self.__str__()


class CNNFCGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('fc_layer_size'),
        StringAttribute('activation'),
        FloatAttribute('input_size'),
        FloatAttribute('weights'),
        FloatAttribute('biases'),
    ]

    @classmethod
    def create(cls, config, input_size):
        key = get_new_global_layer_key()
        gene = cls(key)
        gene.fc_layer_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
        gene.activation = config.activation_function
        gene.input_size = input_size
        return gene

    def initialize_weights(self, config):
        in_size = int(self.input_size)
        fc_size = int(self.fc_layer_size)
        self.weights = np.random.randn(fc_size, in_size) * config.weight_init_stdev + config.weight_init_mean
        self.weights = np.clip(self.weights, config.weight_min_value, config.weight_max_value)
        self.biases = np.random.randn(fc_size) * config.weight_init_stdev + config.weight_init_mean
        self.biases = np.clip(self.biases, config.weight_min_value, config.weight_max_value)

    def mutate(self, config):
        if random() < 0.3:
            new_fc_size = randint(config.fc_layer_size_min, config.fc_layer_size_max)
            self.fc_layer_size = new_fc_size

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"FCGene(fc_size={self.fc_layer_size}, input_size={self.input_size}, activation={self.activation})"

    def __repr__(self):
        return self.__str__()
