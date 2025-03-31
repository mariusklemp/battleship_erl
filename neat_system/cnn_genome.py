import copy
from random import choice, random, randint
from neat.config import ConfigParameter, write_pretty_params

from neat_system.cnn_layers import CNNConvGene, CNNPoolGene, CNNFCGene
from neat_system.helpers import (
    adapt_conv_weights, adapt_biases, adapt_fc_weights,
    calculate_pool_output_size, calculate_conv_output_size,
    compute_gene_type_distance,
    _crossover_by_key)


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
        ConfigParameter("layer_add_prob", float),
        ConfigParameter("layer_delete_prob", float),
        ConfigParameter("compatibility_weight_coefficient", float),
        ConfigParameter("compatibility_disjoint_coefficient", float),
        ConfigParameter("compatibility_excess_coefficient", float),
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
        self.input_keys = [-i - 1 for i in range(self.input_channels * self.input_size ** 2)]
        self.output_keys = [i for i in range(self.output_size)]
        self.pool_type = self.pool_type.split(', ')
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
                    output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
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
        num_fc_layers = randint(0, config.max_num_fc_layer)
        in_channels = config.input_channels
        current_size = config.input_size

        # Create convolutional layers.
        for i in range(num_conv_layers):
            new_gene = CNNConvGene.create(config, in_channels, current_size)
            new_gene.initialize_weights(config)
            new_current_size = ((current_size + 2 * new_gene.padding - new_gene.kernel_size) // new_gene.stride) + 1
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
        flattened_size = max(1, current_size ** 2 * in_channels)
        previous_fc_size = flattened_size
        for i in range(num_fc_layers):
            fc_gene = CNNFCGene.create(config, input_size=previous_fc_size)
            fc_gene.initialize_weights(config)
            self.layer_config.append(fc_gene)
            previous_fc_size = fc_gene.fc_layer_size

        self.enforce_valid_ordering()

    def mutate(self, config):
        """
        Mutate the genome architecture and parameters.
        """
        # Capture a snapshot before any mutation.
        snapshot_before = copy.deepcopy(self)

        # --- Architecture Mutation Phase ---
        if config.mutate_architecture:
            # Possibly add a new convolutional layer.
            if random() < config.layer_add_prob:
                current_input_size = config.input_size
                in_channels = config.input_channels
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = calculate_conv_output_size(current_input_size, gene)
                        in_channels = gene.out_channels
                new_gene = CNNConvGene.create(config, in_channels, current_input_size)
                new_gene.initialize_weights(config)
                fc_indices = [i for i, gene in enumerate(self.layer_config) if isinstance(gene, CNNFCGene)]
                if fc_indices:
                    self.layer_config.insert(fc_indices[0], new_gene)
                else:
                    self.layer_config.append(new_gene)

            # Possibly add a new pooling layer.
            if random() < config.layer_add_prob:
                current_spatial_size = config.input_size
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_spatial_size = calculate_conv_output_size(current_spatial_size, gene)
                    elif isinstance(gene, CNNPoolGene):
                        current_spatial_size = calculate_pool_output_size(current_spatial_size, gene)
                last_layer = self.layer_config[-1]
                if isinstance(last_layer, CNNConvGene):
                    pool_in_channels = last_layer.out_channels
                elif isinstance(last_layer, CNNPoolGene):
                    pool_in_channels = last_layer.in_channels
                else:
                    pool_in_channels = config.input_channels
                new_pool_gene = CNNPoolGene.create(config, pool_in_channels, current_spatial_size)
                self.layer_config.append(new_pool_gene)

            # Possibly add a new fully connected layer.
            if random() < config.layer_add_prob and any(isinstance(g, CNNConvGene) for g in self.layer_config):
                conv_genes = [g for g in self.layer_config if isinstance(g, CNNConvGene)]
                last_conv_gene = conv_genes[-1]
                current_input_size = config.input_size
                for gene in self.layer_config:
                    if isinstance(gene, CNNConvGene):
                        current_input_size = calculate_conv_output_size(current_input_size, gene)
                flattened_size = max(1, current_input_size ** 2 * last_conv_gene.out_channels)
                fc_gene = CNNFCGene.create(config, input_size=flattened_size)
                fc_gene.initialize_weights(config)
                self.layer_config.append(fc_gene)

            # Possibly remove a conv.
            if random() < config.layer_delete_prob:
                active_conv_layers = [gene for gene in self.layer_config if
                                      isinstance(gene, CNNConvGene) and getattr(gene, "enabled", True)]
                if len(active_conv_layers) > 1:
                    gene_to_disable = choice(active_conv_layers)
                    gene_to_disable.enabled = False

            # Possibly remove FC layer.
            if random() < config.layer_delete_prob:
                active_fc_layers = [gene for gene in self.layer_config if
                                    isinstance(gene, CNNFCGene) and getattr(gene, "enabled", True)]
                if active_fc_layers:
                    gene_to_disable = choice(active_fc_layers)
                    gene_to_disable.enabled = False

            # Possibly remove a pooling layer.
            if random() < config.layer_delete_prob:
                active_pool_layers = [g for g in self.layer_config if
                                      isinstance(g, CNNPoolGene) and getattr(g, "enabled", True)]
                if active_pool_layers:
                    gene_to_disable = choice(active_pool_layers)
                    gene_to_disable.enabled = False

        snapshot_arch_done = copy.deepcopy(self)  # Snapshot after architecture mutations
        self.enforce_valid_ordering()
        snapshot_arch_ordering = copy.deepcopy(self)  # Snapshot after architecture mutations
        self._adjust_layer_sizes(config)
        snapshot_arch_done_adjust = copy.deepcopy(self)  # Snapshot after architecture mutations

        # --- Parameter Mutation Phase ---
        new_layer_config = []
        for gene in self.layer_config:
            mutated_gene = gene.mutate(config)
            new_layer_config.append(mutated_gene)
        self.layer_config = new_layer_config

        snapshot_after = copy.deepcopy(self)  # Snapshot after all mutations
        self.enforce_valid_ordering()
        snapshot_after_ordering = copy.deepcopy(self)  # Snapshot after architecture mutations
        self._adjust_layer_sizes(config)
        snapshot_after_adjust = copy.deepcopy(self)  # Snapshot after all mutations

        # Plot all snapshots in one figure.
        genome_steps = [
            (snapshot_before, "Before mutate"),
            (snapshot_arch_done, "After mutate architecture (add, disable)"),
            (snapshot_arch_ordering, "After mutate architecture (reorder)"),
            (snapshot_arch_done_adjust, "After mutate architecture (adjust)"),
            (snapshot_after, "After mutate params"),
            (snapshot_after_ordering, "After mutate params (reorder)"),
            (snapshot_after_adjust, "After mutate params (adjust)")
        ]
        # visualize.plot_multiple_genomes(genome_steps, "Mutation Steps")

    def configure_crossover(self, genome1, genome2, config):
        """
        Perform a NEAT-style crossover between two genomes to produce a new genome.
        If crossover is turned off, simply use the fitter parent.
        """
        # 0) If crossover is off, just clone the fitter parent's genome.
        if not config.crossover_architecture:
            if genome1.fitness >= genome2.fitness:
                self.layer_config = copy.deepcopy(genome1.layer_config)
            else:
                self.layer_config = copy.deepcopy(genome2.layer_config)
            return self

        # 1) Make copies for visualization (optional).
        parent1 = copy.deepcopy(genome1)
        parent2 = copy.deepcopy(genome2)

        # 2) Get parents' fitness.
        fitness1 = genome1.fitness
        fitness2 = genome2.fitness

        # 3) Sort each parent's entire layer_config by innovation number,
        #    rather than splitting by type.
        parent1_genes = sorted(parent1.layer_config, key=lambda g: g.key)
        parent2_genes = sorted(parent2.layer_config, key=lambda g: g.key)

        child_genes = _crossover_by_key(config, parent1_genes, parent2_genes, fitness1, fitness2)

        # 5) Assign to this genome's layer_config.
        self.layer_config = child_genes

        # 6) Ensure we have at least one active conv and FC layer.
        self.ensure_minimum_structure(config)

        # 7) Optionally reorder or validate shapes.
        #    - If you truly want to keep the exact order from the parents,
        #      you might remove or modify enforce_valid_ordering().
        self.enforce_valid_ordering()
        self._adjust_layer_sizes(config)

        # 8) (Optional) Visualize parents & child.
        genome_steps = [
            (parent1, "Parent 1"),
            (parent2, "Parent 2"),
            (self, "Child")
        ]
        # visualize.plot_multiple_genomes(genome_steps, "Crossover Steps")
        return self

    def ensure_minimum_structure(self, config):
        """
        Ensure that the genome has at least one active convolution and one active FC gene.
        If not, either re-enable a disabled gene or add a new gene.
        """
        # Check active convolution genes.
        active_conv = [g for g in self.layer_config if isinstance(g, CNNConvGene) and getattr(g, "enabled", True)]
        if not active_conv:
            # If there is a disabled conv gene, re-enable it.
            disabled_conv = [g for g in self.layer_config if isinstance(g, CNNConvGene)]
            if disabled_conv:
                disabled_conv[0].enabled = True
            else:
                # Optionally, add a new convolution gene.
                new_conv = CNNConvGene.create(config, config.input_channels, config.input_size)
                new_conv.initialize_weights(config)
                self.layer_config.insert(0, new_conv)

    def enforce_valid_ordering(self):
        """
        Reorder *only the enabled* layers so that:
          - All enabled Conv/Pool go before any enabled FC.
          - The first enabled Conv/Pool is a Conv if at all possible.
        Disabled genes remain in the exact same positions.
        """

        # 1) Gather enabled genes separately, preserving order.
        enabled_conv_pool = []
        enabled_fc = []
        for gene in self.layer_config:
            if gene.enabled:
                if isinstance(gene, (CNNConvGene, CNNPoolGene)):
                    enabled_conv_pool.append(gene)
                elif isinstance(gene, CNNFCGene):
                    enabled_fc.append(gene)
                elif isinstance(gene, (CNNConvGene, CNNPoolGene)):
                    enabled_conv_pool.append(gene)

        # 2) Ensure the first enabled Conv/Pool is actually a Conv (if possible).
        if enabled_conv_pool and not isinstance(enabled_conv_pool[0], CNNConvGene):
            for i in range(1, len(enabled_conv_pool)):
                if isinstance(enabled_conv_pool[i], CNNConvGene):
                    # Swap it to the front
                    enabled_conv_pool[0], enabled_conv_pool[i] = enabled_conv_pool[i], enabled_conv_pool[0]
                    break

        # 3) Rebuild the layer_config by replacing enabled genes in place,
        #    but forced into the order: [enabled_conv_pool ... then enabled_fc].
        #    Disabled genes keep their exact positions.
        new_config = []
        for gene in self.layer_config:
            if gene.enabled:
                if enabled_conv_pool:
                    new_config.append(enabled_conv_pool.pop(0))
                else:
                    new_config.append(enabled_fc.pop(0))
            else:
                # Keep disabled genes exactly where they were.
                new_config.append(gene)

        # 4) Assign back
        self.layer_config = new_config

    def _adjust_layer_sizes(self, config):
        # Only consider enabled genes for computing the active network structure.
        active_genes = [gene for gene in self.layer_config if getattr(gene, "enabled", True)]

        current_input_size = config.input_size
        flattened_size = None

        for idx, gene in enumerate(active_genes):
            if isinstance(gene, CNNConvGene):
                if idx == 0:
                    gene.in_channels = config.input_channels
                else:
                    # Find the last enabled convolution gene before this one.
                    previous_convs = [g for g in active_genes[:idx] if isinstance(g, CNNConvGene)]
                    if previous_convs:
                        gene.in_channels = previous_convs[-1].out_channels

                if gene.kernel_size > current_input_size:
                    gene.kernel_size = current_input_size
                try:
                    current_input_size = calculate_conv_output_size(current_input_size, gene)
                except ValueError:
                    gene.kernel_size = max(1, gene.kernel_size - 1)
                    current_input_size = calculate_conv_output_size(current_input_size, gene)
                flattened_size = (current_input_size ** 2) * gene.out_channels

                expected_shape = (int(gene.out_channels), int(gene.in_channels),
                                  int(gene.kernel_size), int(gene.kernel_size))
                if gene.weights.shape != expected_shape:
                    gene.weights = adapt_conv_weights(gene.weights, expected_shape)
                    gene.biases = adapt_biases(gene.biases, expected_shape[0])

            elif isinstance(gene, CNNPoolGene):
                # Update input size first.
                gene.input_size = current_input_size

                # Update in_channels based on the last enabled gene in active_genes.
                if idx > 0:
                    previous = active_genes[idx - 1]
                    if isinstance(previous, (CNNConvGene, CNNPoolGene)):
                        gene.in_channels = previous.out_channels if isinstance(previous,
                                                                               CNNConvGene) else previous.in_channels

                valid_pool_params = config.pre_validate_pool_params(current_input_size)
                if not valid_pool_params:
                    gene.enabled = False
                    continue
                elif (gene.pool_size, gene.stride) not in valid_pool_params:
                    gene.pool_size, gene.stride = valid_pool_params[0]

                new_size = calculate_pool_output_size(current_input_size, gene)
                if new_size <= 0:
                    gene.enabled = False
                    continue
                else:
                    current_input_size = new_size
                flattened_size = (current_input_size ** 2) * gene.in_channels

            elif isinstance(gene, CNNFCGene):
                # For FC genes, use the current flattened size as the input size.
                # If no previous conv/pool layers exist, compute it from the original input.
                if flattened_size is None:
                    # E.g., if there are no conv/pool layers, use input_channels * (input_size)^2.
                    flattened_size = config.input_channels * (config.input_size ** 2)
                gene.input_size = flattened_size
                new_shape = (int(gene.fc_layer_size), int(flattened_size))
                gene.weights = adapt_fc_weights(gene.weights, new_shape)
                gene.biases = adapt_biases(gene.biases, new_shape[0])
                # Update flattened_size to be the output size of this FC gene.
                flattened_size = gene.fc_layer_size

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other.
        The distance is computed separately for convolution, pooling, and fully
        connected genes. For each gene type we distinguish:
          - Excess genes (genes beyond the other genome's maximum innovation).
          - Disjoint genes (non-matching genes within the overlapping range).
          - Matching genes (whose parameter differences are computed via gene.distance()).

        The NEAT formula is applied:
          δ = (c1 * E) / N + (c2 * D) / N + c3 * (average weight difference)
        """
        # -- Convolution Genes --
        conv_self = {g.key: g for g in self.layer_config if isinstance(g, CNNConvGene)}
        conv_other = {g.key: g for g in other.layer_config if isinstance(g, CNNConvGene)}
        conv_distance = compute_gene_type_distance(
            conv_self, conv_other,
            lambda a, b, cfg: a.distance(b, cfg),
            config,
            config.compatibility_excess_coefficient,
            config.compatibility_disjoint_coefficient,
            config.compatibility_weight_coefficient
        )

        # -- Pooling Genes --
        pool_self = {g.key: g for g in self.layer_config if isinstance(g, CNNPoolGene)}
        pool_other = {g.key: g for g in other.layer_config if isinstance(g, CNNPoolGene)}
        pool_distance = compute_gene_type_distance(
            pool_self, pool_other,
            lambda a, b, cfg: a.distance(b, cfg),
            config,
            config.compatibility_excess_coefficient,
            config.compatibility_disjoint_coefficient,
            config.compatibility_weight_coefficient
        )

        # -- Fully Connected Genes --
        fc_self = {g.key: g for g in self.layer_config if isinstance(g, CNNFCGene)}
        fc_other = {g.key: g for g in other.layer_config if isinstance(g, CNNFCGene)}
        fc_distance = compute_gene_type_distance(
            fc_self, fc_other,
            lambda a, b, cfg: a.distance(b, cfg),
            config,
            config.compatibility_excess_coefficient,
            config.compatibility_disjoint_coefficient,
            config.compatibility_weight_coefficient
        )

        return conv_distance + pool_distance + fc_distance

    def size(self):
        """
        Compute and return a summary string of the genome's architecture, including:
          - Total number of layers by type (Convolutional, Pooling, Fully Connected)
          - Total parameter count for trainable layers (Conv and FC)

        :return: Summary string.
        """
        conv_count = sum(1 for gene in self.layer_config if isinstance(gene, CNNConvGene))
        pool_count = sum(1 for gene in self.layer_config if isinstance(gene, CNNPoolGene))
        fc_count = sum(1 for gene in self.layer_config if isinstance(gene, CNNFCGene))
        total_params = 0
        for gene in self.layer_config:
            if isinstance(gene, CNNConvGene):
                params = (gene.kernel_size ** 2) * gene.in_channels * gene.out_channels + gene.out_channels
                total_params += params
            elif isinstance(gene, CNNFCGene):
                params = gene.input_size * gene.fc_layer_size + gene.fc_layer_size
                total_params += params

        return f"Conv: {conv_count}, Pool: {pool_count}, FC: {fc_count}, TTP: {total_params}"
