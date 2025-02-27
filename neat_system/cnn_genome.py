import copy
import time
from random import choice, random, randint
from neat.config import ConfigParameter, write_pretty_params

from neat_system.cnn_layers import CNNConvGene, CNNPoolGene, CNNFCGene
from neat_system.helpers import (
    graph_architecture_distance,
    connection_distance,
    adapt_conv_weights,
    adapt_biases,
    adapt_fc_weights,
    _crossover_vector,
    _crossover_matrix,
    _mutate_array,
    calculate_pool_output_size,
    calculate_conv_output_size,
    _mutate_weights_conv,
    _mutate_weights_fc,
)


class CNNGenomeConfig(object):
    """
    Configuration for a CNN genome. This object holds hyperparameters that control
    the structure and evolution of convolutional, pooling, and fully connected layers.
    """

    __params = [
        ConfigParameter("input_channels", int),
        ConfigParameter("input_size", int),
        ConfigParameter("output_size", int),
        ConfigParameter("out_channels_min", int),
        ConfigParameter("out_channels_max", int),
        ConfigParameter("max_num_conv_layer", int),
        ConfigParameter("max_num_fc_layer", int),
        ConfigParameter("kernel_sizes", str),
        ConfigParameter("strides", str),
        ConfigParameter("paddings", str),
        ConfigParameter("activation_function", str),
        ConfigParameter("fc_layer_size_min", int),
        ConfigParameter("fc_layer_size_max", int),
        ConfigParameter("pool_sizes", str),
        ConfigParameter("pool_strides", str),
        ConfigParameter("pool_type", str),
        ConfigParameter("max_num_pool_layer", int),
        ConfigParameter("pool_layer_add_prob", float),
        ConfigParameter("pool_layer_delete_prob", float),
        ConfigParameter("layer_add_prob", float),
        ConfigParameter("layer_delete_prob", float),
        ConfigParameter("conv_params_mutate_prob", float),
        ConfigParameter("conv_output_mutate_prob", float),
        ConfigParameter("architecture_mutation_prob", float),
        ConfigParameter("compatibility_topology_coefficient", float),
        ConfigParameter("compatibility_weight_coefficient", float),
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
        ConfigParameter("crossover_weights", bool),
    ]

    def __init__(self, params):
        """
        Initialize the configuration from a dictionary of parameters.

        :param params: Dictionary containing configuration parameters.
        """
        for param in self.__params:
            setattr(self, param.name, param.interpret(params))
        self.kernel_sizes = eval(self.kernel_sizes)
        self.strides = eval(self.strides)
        self.paddings = eval(self.paddings)
        self.valid_conv_params = self.pre_validate_conv_params(self.input_size)
        self.input_keys = [
            -i - 1 for i in range(self.input_channels * self.input_size**2)
        ]
        self.output_keys = [i for i in range(self.output_size)]
        self.pool_type = self.pool_type.split(", ")
        self.pool_sizes = eval(self.pool_sizes)
        self.pool_strides = eval(self.pool_strides)

    def pre_validate_conv_params(self, input_size):
        """
        Validate convolution parameters for a given input size.

        :param input_size: The spatial dimension of the input.
        :return: A list of valid (kernel_size, stride, padding) tuples.
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

    def pre_validate_pool_params(self, input_size):
        """
        Validate pooling parameters for a given input size.

        :param input_size: The spatial dimension of the input.
        :return: A list of valid (pool_size, stride) tuples.
        """
        valid_pool_params = []
        for pool_size in self.pool_sizes:
            if input_size < pool_size:
                continue
            for stride in self.pool_strides:
                output_size = ((input_size - pool_size) // stride) + 1
                if output_size > 0:
                    valid_pool_params.append((pool_size, stride))
        return valid_pool_params

    def save(self, f):
        """
        Save configuration parameters to a file in a pretty format.

        :param f: File-like object to write the configuration.
        """
        write_pretty_params(f, self, self.__params)


class CNNGenome(object):
    """
    Represents a CNN genome composed of an ordered list of layer genes (convolutional,
    pooling, and fully connected). The genome also maintains a fitness value.
    """

    @classmethod
    def parse_config(cls, param_dict):
        """
        Parse the configuration parameters and return a CNNGenomeConfig instance.

        :param param_dict: Dictionary of configuration parameters.
        :return: CNNGenomeConfig instance.
        """
        return CNNGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        """
        Write the configuration to a file.

        :param f: File-like object.
        :param config: CNNGenomeConfig instance.
        """
        config.save(f)

    def __init__(self, key):
        """
        Initialize a new CNN genome.

        :param key: Unique identifier for the genome.
        """
        self.key = key
        self.layer_config = []  # List of layer genes.
        self.fitness = None

    def configure_new(self, config):
        """
        Configure a new genome with a random architecture and initialize weights.

        :param config: CNNGenomeConfig instance.
        """
        self.layer_config = []
        num_conv_layers = randint(1, config.max_num_conv_layer)
        num_pool_layers = randint(0, config.max_num_pool_layer)
        num_fc_layers = randint(1, config.max_num_fc_layer)
        in_channels = config.input_channels
        current_size = config.input_size

        # Create convolutional layers.
        for i in range(num_conv_layers):
            new_gene = CNNConvGene.create(config, in_channels, current_size)
            new_gene.initialize_weights(config)
            new_current_size = (
                (current_size + 2 * new_gene.padding - new_gene.kernel_size)
                // new_gene.stride
            ) + 1
            self.layer_config.append(new_gene)
            in_channels = new_gene.out_channels
            current_size = new_current_size

            # Optionally add a pooling layer.
            if num_pool_layers > 0 and random() < 0.5:
                pool_gene = CNNPoolGene.create(config, in_channels, current_size)
                self.layer_config.append(pool_gene)
                current_size = calculate_pool_output_size(current_size, pool_gene)
                num_pool_layers -= 1

        # Create fully connected layers.
        flattened_size = max(1, current_size**2 * in_channels)
        previous_fc_size = flattened_size
        for i in range(num_fc_layers):
            fc_gene = CNNFCGene.create(config, input_size=previous_fc_size)
            fc_gene.initialize_weights(config)
            self.layer_config.append(fc_gene)
            previous_fc_size = fc_gene.fc_layer_size

        self.enforce_valid_ordering()

    def mutate(self, config):
        """
        Mutate the genome architecture and parameters. This may involve adding or removing layers
        and mutating each layer gene.

        :param config: CNNGenomeConfig instance.
        """
        # print('DOING MUTATE')
        # print(self.layer_config)
        # --- Architecture Mutation Phase ---
        if config.mutate_architecture:
            # Possibly add a new convolutional layer.
            if random() < config.layer_add_prob:
                # print('ADDING CONV')
                current_input_size = config.input_size
                in_channels = config.input_channels
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = calculate_conv_output_size(
                            current_input_size, gene
                        )
                        in_channels = gene.out_channels
                new_gene = CNNConvGene.create(config, in_channels, current_input_size)
                new_gene.initialize_weights(config)
                fc_indices = [
                    i
                    for i, gene in enumerate(self.layer_config)
                    if isinstance(gene, CNNFCGene)
                ]
                if fc_indices:
                    self.layer_config.insert(fc_indices[0], new_gene)
                else:
                    self.layer_config.append(new_gene)

            # Possibly add a new pooling layer.
            if random() < config.pool_layer_add_prob:
                # print('ADDING POOL')
                current_spatial_size = config.input_size
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_spatial_size = calculate_conv_output_size(
                            current_spatial_size, gene
                        )
                    elif isinstance(gene, CNNPoolGene):
                        current_spatial_size = calculate_pool_output_size(
                            current_spatial_size, gene
                        )
                last_layer = self.layer_config[-1]
                if isinstance(last_layer, CNNConvGene):
                    pool_in_channels = last_layer.out_channels
                elif isinstance(last_layer, CNNPoolGene):
                    pool_in_channels = last_layer.in_channels
                else:
                    pool_in_channels = config.input_channels
                new_pool_gene = CNNPoolGene.create(
                    config, pool_in_channels, current_spatial_size
                )
                self.layer_config.append(new_pool_gene)

            # Possibly add a new fully connected layer.
            if random() < config.layer_add_prob and any(
                isinstance(g, CNNConvGene) for g in self.layer_config
            ):
                # print('ADDING FC')
                conv_genes = [
                    g for g in self.layer_config if isinstance(g, CNNConvGene)
                ]
                last_conv_gene = conv_genes[-1]
                current_input_size = config.input_size
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = calculate_conv_output_size(
                            current_input_size, gene
                        )
                flattened_size = max(
                    1, current_input_size**2 * last_conv_gene.out_channels
                )
                fc_gene = CNNFCGene.create(config, input_size=flattened_size)
                fc_gene.initialize_weights(config)
                self.layer_config.append(fc_gene)

            # Possibly remove a layer.
            if random() < config.layer_delete_prob and len(self.layer_config) > 1:
                fc_genes = [
                    gene for gene in self.layer_config if isinstance(gene, CNNFCGene)
                ]
                if len(fc_genes) > 1 or not isinstance(
                    self.layer_config[-1], CNNFCGene
                ):
                    # print('DELETING')
                    self.layer_config.pop()

            # Possibly remove a pooling layer.
            if random() < config.pool_layer_delete_prob:
                pool_layers = [
                    g for g in self.layer_config if isinstance(g, CNNPoolGene)
                ]
                if pool_layers:
                    # print('DELETING POOL')
                    self.layer_config.remove(choice(pool_layers))

            # print('BEFORE adjusting layer sizes')
            self.enforce_valid_ordering()
            # print(self.layer_config)
            self._adjust_layer_sizes(config)
            # print('AFTER adjusting layer sizes')
            # print(self.layer_config)

            # --- Mutate each gene ---
            for gene in self.layer_config:
                self._adjust_layer_sizes(config)
                gene.mutate(config)

        # --- Weight Mutation Phase (if enabled) ---
        if config.mutate_weights:
            for i, gene in enumerate(self.layer_config):
                if isinstance(gene, CNNConvGene):
                    _mutate_weights_conv(
                        gene,
                        config.weight_mutate_rate,
                        config.weight_mutate_power,
                        config.weight_replace_rate,
                        config.weight_min_value,
                        config.weight_max_value,
                        config,
                    )
                elif isinstance(gene, CNNFCGene):
                    _mutate_weights_fc(
                        gene,
                        config.weight_mutate_rate,
                        config.weight_mutate_power,
                        config.weight_replace_rate,
                        config.weight_min_value,
                        config.weight_max_value,
                        config,
                    )

        self.enforce_valid_ordering()
        self._adjust_layer_sizes(config)
        # print('AFTER MUTATE')
        # print(self.layer_config)

    def configure_crossover(self, genome1, genome2, config):
        """
        Perform crossover between two genomes to produce a new genome.

        :param genome1: Parent genome 1.
        :param genome2: Parent genome 2.
        :param config: CNNGenomeConfig instance.
        """
        start = time.perf_counter()

        # print('DOING CROSSOVER')
        # print(genome1.layer_config)
        # print(genome2.layer_config)

        conv_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNConvGene)]
        conv_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNConvGene)]
        pool_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNPoolGene)]
        pool_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNPoolGene)]
        fc_layers_1 = [g for g in genome1.layer_config if isinstance(g, CNNFCGene)]
        fc_layers_2 = [g for g in genome2.layer_config if isinstance(g, CNNFCGene)]

        if config.crossover_architecture:
            self.layer_config = []
            in_channels = config.input_channels
            current_input_size = config.input_size

            # Step 1: Crossover convolutional layers.
            new_conv_layers = []
            for i, (g1, g2) in enumerate(zip(conv_layers_1, conv_layers_2)):
                selected = g1.copy() if random() < 0.5 else g2.copy()
                selected.in_channels = in_channels
                while (
                    selected.kernel_size > current_input_size
                    and selected.kernel_size > 1
                ):
                    selected.kernel_size -= 1
                try:
                    current_input_size = calculate_conv_output_size(
                        current_input_size, selected
                    )
                    in_channels = selected.out_channels
                    new_conv_layers.append(selected)
                except ValueError as e:
                    print(f"[ERROR] Crossover Conv layer adjustment failed: {e}")

            # Step 2: Crossover pooling layers (after conv layers).
            new_pool_layers = []
            for g1, g2 in zip(pool_layers_1, pool_layers_2):
                selected = g1.copy() if random() < 0.5 else g2.copy()
                if selected.pool_size <= current_input_size:
                    current_input_size = calculate_pool_output_size(
                        current_input_size, selected
                    )
                    new_pool_layers.append(selected)

            # Step 3: Append conv and pool layers.
            self.layer_config.extend(new_conv_layers)
            self.layer_config.extend(new_pool_layers)

            # Step 4: Determine flattened size after last conv/pool layer.
            flattened_size = (
                max(1, current_input_size**2 * in_channels)
                if self.layer_config
                else None
            )

            # Step 5: Crossover fully connected layers.
            new_fc_layers = []
            for i, (g1, g2) in enumerate(zip(fc_layers_1, fc_layers_2)):
                selected = g1.copy() if random() < 0.5 else g2.copy()
                if i == 0 and flattened_size is not None:
                    selected.input_size = flattened_size
                new_fc_layers.append(selected)

            # Step 6: Append FC layers.
            self.layer_config.extend(new_fc_layers)
        else:
            chosen = genome1 if random() < 0.5 else genome2
            self.layer_config = copy.deepcopy(chosen.layer_config)

        if config.crossover_weights:
            conv_index = 0
            fc_index = 0
            for i, gene in enumerate(self.layer_config):
                if isinstance(gene, CNNConvGene):
                    try:
                        parent_gene = (
                            conv_layers_1[conv_index]
                            if random() < 0.5
                            else conv_layers_2[conv_index]
                        )
                    except IndexError:
                        parent_gene = (
                            conv_layers_1[-1] if conv_layers_1 else conv_layers_2[-1]
                        )
                    conv_index += 1
                    target_weight_shape = (
                        int(gene.out_channels),
                        int(gene.in_channels),
                        int(gene.kernel_size),
                        int(gene.kernel_size),
                    )
                    gene.weights = _crossover_matrix(
                        gene.weights,
                        parent_gene.weights,
                        adapt_conv_weights,
                        target_weight_shape,
                    )
                    gene.biases = _crossover_vector(
                        gene.biases, parent_gene.biases, target_weight_shape[0]
                    )
                elif isinstance(gene, CNNFCGene):
                    try:
                        parent_gene = (
                            fc_layers_1[fc_index]
                            if random() < 0.5
                            else fc_layers_2[fc_index]
                        )
                    except IndexError:
                        parent_gene = (
                            fc_layers_1[-1] if fc_layers_1 else fc_layers_2[-1]
                        )
                    fc_index += 1
                    target_weight_shape = (
                        int(gene.fc_layer_size),
                        int(gene.input_size),
                    )
                    gene.weights = _crossover_matrix(
                        gene.weights,
                        parent_gene.weights,
                        adapt_fc_weights,
                        target_weight_shape,
                    )
                    gene.biases = _crossover_vector(
                        gene.biases, parent_gene.biases, target_weight_shape[0]
                    )

        # print('AFTER CROSSOVER')
        self.enforce_valid_ordering()
        # print('BEFORE adjusting layer sizes')
        # print(self.layer_config)
        self._adjust_layer_sizes(config)
        # print('AFTER adjusting layer sizes')
        # print(self.layer_config)

    def enforce_valid_ordering(self):
        """
        Reorder the layer configuration so that convolutional and pooling layers come before
        fully connected layers.
        """
        conv_pool_genes = [
            gene
            for gene in self.layer_config
            if isinstance(gene, (CNNConvGene, CNNPoolGene))
        ]
        fc_genes = [gene for gene in self.layer_config if isinstance(gene, CNNFCGene)]
        self.layer_config = conv_pool_genes + fc_genes

    def _adjust_layer_sizes(self, config):
        """
        Adjust the spatial dimensions and flattened size throughout the network.
        This method updates layer parameters and adapts weights as necessary.
        """
        current_input_size = config.input_size
        flattened_size = None

        for i, gene in enumerate(self.layer_config):
            if isinstance(gene, CNNConvGene):
                if i == 0:
                    gene.in_channels = config.input_channels
                else:
                    conv_genes_before = [
                        g for g in self.layer_config[:i] if isinstance(g, CNNConvGene)
                    ]
                    if conv_genes_before:
                        gene.in_channels = conv_genes_before[-1].out_channels

                if gene.kernel_size > current_input_size:
                    gene.kernel_size = current_input_size
                try:
                    current_input_size = calculate_conv_output_size(
                        current_input_size, gene
                    )
                except ValueError as e:
                    gene.kernel_size = max(1, gene.kernel_size - 1)
                    current_input_size = calculate_conv_output_size(
                        current_input_size, gene
                    )
                flattened_size = (current_input_size**2) * gene.out_channels

                expected_shape = (
                    int(gene.out_channels),
                    int(gene.in_channels),
                    int(gene.kernel_size),
                    int(gene.kernel_size),
                )
                if gene.weights.shape != expected_shape:
                    gene.weights = adapt_conv_weights(gene.weights, expected_shape)
                    gene.biases = adapt_biases(gene.biases, expected_shape[0])

            elif isinstance(gene, CNNPoolGene):
                # Update the pooling gene's input size based on the current spatial dimension.
                gene.input_size = current_input_size
                if i > 0:
                    prev_gene = self.layer_config[i - 1]
                    if isinstance(prev_gene, CNNConvGene):
                        gene.in_channels = prev_gene.out_channels
                    elif isinstance(prev_gene, CNNPoolGene):
                        gene.in_channels = prev_gene.in_channels
                current_input_size = calculate_pool_output_size(
                    current_input_size, gene
                )
                flattened_size = (current_input_size**2) * gene.in_channels

            elif isinstance(gene, CNNFCGene):
                new_shape = (int(gene.fc_layer_size), int(flattened_size))
                gene.weights = adapt_fc_weights(gene.weights, new_shape)
                gene.biases = adapt_biases(gene.biases, new_shape[0])
                gene.input_size = flattened_size
                flattened_size = gene.fc_layer_size

    def distance(self, other, config):
        """
        Compute a composite distance between two CNN genomes as a weighted sum of:
          1. Graph-based architecture distance.
          2. Connection (weight and bias) distance.

        :param other: Another CNNGenome instance.
        :param config: CNNGenomeConfig instance with distance coefficients.
        :return: Composite distance (float).
        """
        arch_distance = graph_architecture_distance(self, other)
        arch_distance = arch_distance * config.compatibility_topology_coefficient
        conn_distance_val = connection_distance(self, other)
        conn_distance_val = conn_distance_val * config.compatibility_weight_coefficient
        combined_distance = arch_distance + conn_distance_val
        return combined_distance

    def size(self):
        """
        Compute and return a summary string of the genome's architecture, including:
          - Total number of layers by type (Convolutional, Pooling, Fully Connected)
          - Total parameter count for trainable layers (Conv and FC)

        :return: Summary string.
        """
        conv_count = sum(
            1 for gene in self.layer_config if isinstance(gene, CNNConvGene)
        )
        pool_count = sum(
            1 for gene in self.layer_config if isinstance(gene, CNNPoolGene)
        )
        fc_count = sum(1 for gene in self.layer_config if isinstance(gene, CNNFCGene))
        total_params = 0
        for gene in self.layer_config:
            if isinstance(gene, CNNConvGene):
                params = (
                    gene.kernel_size**2
                ) * gene.in_channels * gene.out_channels + gene.out_channels
                total_params += params
            elif isinstance(gene, CNNFCGene):
                params = gene.input_size * gene.fc_layer_size + gene.fc_layer_size
                total_params += params

        return f"Conv: {conv_count}, Pool: {pool_count}, FC: {fc_count}, TTP: {total_params}"
