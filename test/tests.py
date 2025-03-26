import numpy as np
from matplotlib import pyplot as plt

import visualize
# Adjust the module paths as needed.
from neat_system.cnn_genome import CNNGenome, CNNGenomeConfig
from neat_system.cnn_layers import CNNFCGene, CNNConvGene, CNNPoolGene


def create_manual_genomes():
    """
    Create two genomes with:
      - Genome 1: [ConvGene(innov=0), ConvGene(innov=1), FCGene(innov=2)]
      - Genome 2: [ConvGene(innov=0), PoolGene(innov=3), ConvGene(innov=1), FCGene(innov=2)]
    """
    # --- Genome 1 ---
    genome1 = CNNGenome(key=1)

    # ConvGene with innov=0.
    conv_gene1 = CNNConvGene(0)
    conv_gene1.kernel_size = 3
    conv_gene1.stride = 1
    conv_gene1.padding = 0
    conv_gene1.input_size = 3
    conv_gene1.in_channels = 4
    conv_gene1.out_channels = 32
    conv_gene1.activation = "relu"
    conv_gene1.enabled = True
    conv_gene1.weights = np.random.randn(32, 4, 3, 3)
    conv_gene1.biases = np.random.randn(32)

    # ConvGene with innov=1.
    conv_gene2 = CNNConvGene(3)
    conv_gene2.kernel_size = 1
    conv_gene2.stride = 1
    conv_gene2.padding = 0
    conv_gene2.input_size = 1
    conv_gene2.in_channels = 32
    conv_gene2.out_channels = 64
    conv_gene2.activation = "relu"
    conv_gene2.enabled = True
    conv_gene2.weights = np.random.randn(64, 32, 1, 1)
    conv_gene2.biases = np.random.randn(64)

    # FCGene with innov=2.
    fc_gene1 = CNNFCGene(2)
    fc_gene1.fc_layer_size = 128
    fc_gene1.input_size = 64
    fc_gene1.activation = "relu"
    fc_gene1.enabled = True
    fc_gene1.weights = np.random.randn(128, 64)
    fc_gene1.biases = np.random.randn(128)

    genome1.layer_config = [conv_gene1, conv_gene2, fc_gene1]

    # --- Genome 2 ---
    genome2 = CNNGenome(key=2)

    # ConvGene with innov=0.
    conv_gene1_b = CNNConvGene(0)
    conv_gene1_b.kernel_size = 3
    conv_gene1_b.stride = 1
    conv_gene1_b.padding = 0
    conv_gene1_b.input_size = 3
    conv_gene1_b.in_channels = 4
    conv_gene1_b.out_channels = 40  # Different parameter.
    conv_gene1_b.activation = "relu"
    conv_gene1_b.enabled = True
    conv_gene1_b.weights = np.random.randn(40, 4, 3, 3)
    conv_gene1_b.biases = np.random.randn(40)

    # PoolGene with innov=3 (extra gene in Genome 2).
    pool_gene = CNNPoolGene(1)
    pool_gene.pool_size = 2
    pool_gene.stride = 1
    pool_gene.pool_type = "avg"
    pool_gene.input_size = 1
    pool_gene.in_channels = 40
    pool_gene.enabled = True

    # ConvGene with innov=1.
    conv_gene2_b = CNNConvGene(3)
    conv_gene2_b.kernel_size = 1
    conv_gene2_b.stride = 1
    conv_gene2_b.padding = 0
    conv_gene2_b.input_size = 1
    conv_gene2_b.in_channels = 40
    conv_gene2_b.out_channels = 64
    conv_gene2_b.activation = "relu"
    conv_gene2_b.enabled = True
    conv_gene2_b.weights = np.random.randn(64, 40, 1, 1)
    conv_gene2_b.biases = np.random.randn(64)

    # FCGene with innov=2.
    fc_gene2 = CNNFCGene(2)
    fc_gene2.fc_layer_size = 128
    fc_gene2.input_size = 64
    fc_gene2.activation = "relu"
    fc_gene2.enabled = True
    fc_gene2.weights = np.random.randn(128, 64)
    fc_gene2.biases = np.random.randn(128)

    genome2.layer_config = [conv_gene1_b, pool_gene, conv_gene2_b, fc_gene2]

    return genome1, genome2


def create_manual_genomes_with_pool_first():
    """
    Create two genomes with genes sorted by innovation number:
      - Both genomes: [PoolGene(innov=0), ConvGene(innov=1), FCGene(innov=2)]
    """
    # --- Genome 1 ---
    genome1 = CNNGenome(key=1)

    # PoolGene with innov=0.
    pool_gene1 = CNNPoolGene(0)
    pool_gene1.pool_size = 2
    pool_gene1.stride = 1
    pool_gene1.pool_type = "max"
    pool_gene1.input_size = 3
    pool_gene1.in_channels = 4
    pool_gene1.enabled = True
    pool_gene1.weights = np.random.randn(4, 4)
    pool_gene1.biases = np.random.randn(4)

    # ConvGene with innov=1.
    conv_gene1 = CNNConvGene(1)
    conv_gene1.kernel_size = 3
    conv_gene1.stride = 1
    conv_gene1.padding = 0
    conv_gene1.input_size = 3
    conv_gene1.in_channels = 4
    conv_gene1.out_channels = 32
    conv_gene1.activation = "relu"
    conv_gene1.enabled = True
    conv_gene1.weights = np.random.randn(32, 4, 3, 3)
    conv_gene1.biases = np.random.randn(32)

    # FCGene with innov=2.
    fc_gene1 = CNNFCGene(2)
    fc_gene1.fc_layer_size = 128
    fc_gene1.input_size = 32
    fc_gene1.activation = "relu"
    fc_gene1.enabled = True
    fc_gene1.weights = np.random.randn(128, 32)
    fc_gene1.biases = np.random.randn(128)

    genome1.layer_config = [pool_gene1, conv_gene1, fc_gene1]

    # --- Genome 2 ---
    genome2 = CNNGenome(key=2)

    # PoolGene with innov=0.
    pool_gene2 = CNNPoolGene(0)
    pool_gene2.pool_size = 2
    pool_gene2.stride = 1
    pool_gene2.pool_type = "max"
    pool_gene2.input_size = 3
    pool_gene2.in_channels = 4
    pool_gene2.enabled = True
    pool_gene2.weights = np.random.randn(4, 4)
    pool_gene2.biases = np.random.randn(4)

    # ConvGene with innov=1 (different parameter).
    conv_gene2 = CNNConvGene(1)
    conv_gene2.kernel_size = 3
    conv_gene2.stride = 1
    conv_gene2.padding = 0
    conv_gene2.input_size = 3
    conv_gene2.in_channels = 4
    conv_gene2.out_channels = 40
    conv_gene2.activation = "relu"
    conv_gene2.enabled = True
    conv_gene2.weights = np.random.randn(40, 4, 3, 3)
    conv_gene2.biases = np.random.randn(40)

    # FCGene with innov=2.
    fc_gene2 = CNNFCGene(2)
    fc_gene2.fc_layer_size = 128
    fc_gene2.input_size = 40
    fc_gene2.activation = "relu"
    fc_gene2.enabled = True
    fc_gene2.weights = np.random.randn(128, 40)
    fc_gene2.biases = np.random.randn(128)

    genome2.layer_config = [pool_gene2, conv_gene2, fc_gene2]

    return genome1, genome2


def test_crossover_variability():
    """
    Test the unified NEAT-style crossover on genomes where the pool gene is not first.
    Genome 1: [Conv(0), Conv(1), FC(2)]
    Genome 2: [Conv(0), Pool(3), Conv(1), FC(2)]
    """
    config_params = {
        "input_channels": 4,
        "input_size": 3,
        "output_size": 9,
        "kernel_sizes": "[3, 1]",  # Simplified.
        "strides": "[1]",
        "paddings": "[0]",
        "out_channels_min": 32,
        "out_channels_max": 64,
        "fc_layer_size_min": 32,
        "fc_layer_size_max": 256,
        "compatibility_topology_coefficient": 0.1,
        "compatibility_weight_coefficient": 1,
        "compatibility_disjoint_coefficient": 1,
        "compatibility_excess_coefficient": 1,
        "activation_function": "relu",
        "max_num_conv_layer": 2,
        "max_num_fc_layer": 1,
        "architecture_mutation_prob": 0.3,
        "layer_add_prob": 0.1,
        "layer_delete_prob": 0.1,
        "conv_params_mutate_prob": 0.1,
        "pool_params_mutate_prob": 0.1,
        "fc_params_mutate_prob": 0.1,
        "pool_sizes": "[2]",
        "pool_strides": "[1]",
        "pool_type": "max, avg",
        "max_num_pool_layer": 1,
        "pool_layer_add_prob": 0.2,
        "pool_layer_delete_prob": 0.2,
        "weight_init_mean": 0.0,
        "weight_init_stdev": 0.1,
        "weight_init_type": "gaussian",
        "weight_max_value": 5.0,
        "weight_min_value": -5.0,
        "weight_mutate_rate": 0.3,
        "weight_mutate_power": 0.3,
        "weight_replace_rate": 0.3,
        "mutate_architecture": "True",
        "mutate_weights": "True",
        "crossover_architecture": "True",
        "crossover_weights": "False",
    }
    config = CNNGenomeConfig(config_params)

    genome1, genome2 = create_manual_genomes()

    print("=== Test A: Parent Genomes (Unsorted Order) ===")
    print("Genome 1:")
    for gene in genome1.layer_config:
        print(gene)
    print("\nGenome 2:")
    for gene in genome2.layer_config:
        print(gene)

    # Set fitness: Genome1 fitter.
    genome1.fitness = 10.0
    genome2.fitness = 8.0

    print("\n=== Crossover Variability (Genome1 fitter) ===")
    for i in range(3):
        offspring = CNNGenome(key=100 + i)
        offspring.configure_crossover(genome1, genome2, config)
        print(f"\n--- Offspring {i + 1} (Genome1 fitter) ---")
        for gene in offspring.layer_config:
            print(gene)

    # Now test with Genome2 fitter.
    genome1.fitness = 8.0
    genome2.fitness = 10.0

    print("\n=== Crossover Variability (Genome2 fitter) ===")
    for i in range(3):
        offspring = CNNGenome(key=200 + i)
        offspring.configure_crossover(genome1, genome2, config)
        print(f"\n--- Offspring {i + 1} (Genome2 fitter) ---")
        for gene in offspring.layer_config:
            print(gene)


def test_crossover_pool_first():
    """
    Test the unified NEAT-style crossover on genomes where the PoolGene
    has the lowest innovation number. Both genomes: [Pool(0), Conv(1), FC(2)].
    This verifies that the ordering is preserved.
    """
    config_params = {
        "input_channels": 4,
        "input_size": 3,
        "output_size": 9,
        "kernel_sizes": "[3]",
        "strides": "[1]",
        "paddings": "[0]",
        "out_channels_min": 32,
        "out_channels_max": 64,
        "fc_layer_size_min": 32,
        "fc_layer_size_max": 256,
        "compatibility_topology_coefficient": 0.1,
        "compatibility_weight_coefficient": 1,
        "compatibility_disjoint_coefficient": 1,
        "compatibility_excess_coefficient": 1,
        "activation_function": "relu",
        "max_num_conv_layer": 1,
        "max_num_fc_layer": 1,
        "architecture_mutation_prob": 0.3,
        "layer_add_prob": 0.1,
        "layer_delete_prob": 0.1,
        "conv_params_mutate_prob": 0.1,
        "pool_params_mutate_prob": 0.1,
        "fc_params_mutate_prob": 0.1,
        "pool_sizes": "[2]",
        "pool_strides": "[1]",
        "pool_type": "max",
        "max_num_pool_layer": 1,
        "pool_layer_add_prob": 0.2,
        "pool_layer_delete_prob": 0.2,
        "weight_init_mean": 0.0,
        "weight_init_stdev": 0.1,
        "weight_init_type": "gaussian",
        "weight_max_value": 5.0,
        "weight_min_value": -5.0,
        "weight_mutate_rate": 0.3,
        "weight_mutate_power": 0.3,
        "weight_replace_rate": 0.3,
        "mutate_architecture": "True",
        "mutate_weights": "True",
        "crossover_architecture": "True",
        "crossover_weights": "False",
    }
    config = CNNGenomeConfig(config_params)

    genome1, genome2 = create_manual_genomes_with_pool_first()

    print("\n=== Test B: Parent Genomes (PoolGene First) ===")
    print("Genome 1:")
    for gene in genome1.layer_config:
        print(gene)
    print("\nGenome 2:")
    for gene in genome2.layer_config:
        print(gene)

    # Set equal fitness.
    genome1.fitness = 10.0
    genome2.fitness = 10.0

    # Perform a single crossover.
    offspring = CNNGenome(key=300)
    offspring.configure_crossover(genome1, genome2, config)

    print("\n=== Offspring Genome (PoolGene First) ===")
    for gene in offspring.layer_config:
        print(gene)


def create_manual_genomes_vis():
    """
    Create two genomes with:
      - Genome 1: [ConvGene(innov=0), ConvGene(innov=1), FCGene(innov=2)]
      - Genome 2: [ConvGene(innov=0), PoolGene(innov=3), ConvGene(innov=1), FCGene(innov=2)]
    """
    # --- Genome 1 ---
    genome1 = CNNGenome(key=1)

    # ConvGene with innov=0.
    conv_gene1 = CNNConvGene(0)
    conv_gene1.kernel_size = 3
    conv_gene1.stride = 1
    conv_gene1.padding = 0
    conv_gene1.input_size = 3
    conv_gene1.in_channels = 4
    conv_gene1.out_channels = 32
    conv_gene1.activation = "relu"
    conv_gene1.enabled = False
    conv_gene1.weights = np.random.randn(32, 4, 3, 3)
    conv_gene1.biases = np.random.randn(32)

    # ConvGene with innov=1.
    conv_gene2 = CNNConvGene(3)
    conv_gene2.kernel_size = 1
    conv_gene2.stride = 1
    conv_gene2.padding = 0
    conv_gene2.input_size = 1
    conv_gene2.in_channels = 32
    conv_gene2.out_channels = 64
    conv_gene2.activation = "relu"
    conv_gene2.enabled = True
    conv_gene2.weights = np.random.randn(64, 32, 1, 1)
    conv_gene2.biases = np.random.randn(64)

    # PoolGene with innov=3 (extra gene in Genome 2).
    pool_gene = CNNPoolGene(7)
    pool_gene.pool_size = 2
    pool_gene.stride = 1
    pool_gene.pool_type = "avg"
    pool_gene.input_size = 1
    pool_gene.in_channels = 40
    pool_gene.enabled = True

    # FCGene with innov=2.
    fc_gene1 = CNNFCGene(4)
    fc_gene1.fc_layer_size = 128
    fc_gene1.input_size = 64
    fc_gene1.activation = "relu"
    fc_gene1.enabled = True
    fc_gene1.weights = np.random.randn(128, 64)
    fc_gene1.biases = np.random.randn(128)

    genome1.layer_config = [conv_gene1, conv_gene2, pool_gene, fc_gene1]

    return genome1


if __name__ == '__main__':
    # Run both tests.
    #test_crossover_variability()
    #test_crossover_pool_first()
    # Create the figure and axes
    genome = create_manual_genomes_vis()
    visualize.visualize_genome_simple(genome,"Genome structure")
    visualize.plot_genome_simple(genome, "Genome structure")
