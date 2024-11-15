import random
from neat.genome import DefaultGenome


class CNNGenome(DefaultGenome):
    """
    A custom genome class to store and evolve CNN-specific parameters.
    """

    def __init__(self, key):
        super().__init__(key)
        self.layer_config = []  # Store layer configurations
        self.board_size = 5

    def configure_new(self, config):
        """
        Initialize a new genome with random CNN layer configurations while ensuring valid output sizes.
        """
        super().configure_new(config)
        self.layer_config = []
        current_size = self.board_size
        num_layers = random.randint(3, 5)   # Enforce a minimum of 3 layers

        added_fc_layer = False

        for _ in range(num_layers):
            # Enforce Conv layers at the start
            if not added_fc_layer and random.random() < 0.5:
                kernel_size = min(random.choice([1, 3, 5]), current_size)
                stride = min(random.randint(1, 2), current_size)
                padding = random.randint(0, (kernel_size - 1) // 2)

                output_size = ((current_size - kernel_size + 2 * padding) // stride) + 1
                if output_size <= 0:
                    continue  # Skip invalid layers

                self.layer_config.append({
                    'layer_type': 'conv',
                    'out_channels': random.randint(16, 48),
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                })
                current_size = output_size
            else:
                if current_size > 1 and not added_fc_layer:  # Add FC layers only after Conv layers
                    self.layer_config.append({
                        'layer_type': 'fc',
                        'fc_layer_size': random.randint(32, 128),
                    })
                    added_fc_layer = True

        # Enforce at least one Conv layer
        if not any(layer['layer_type'] == 'conv' for layer in self.layer_config):
            self.layer_config.insert(0, {
                'layer_type': 'conv',
                'out_channels': 16,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
            })

    def configure_crossover(self, genome1, genome2, config):
        """
        Combine layer configurations from two parent genomes.
        """
        super().configure_crossover(genome1, genome2, config)
        self.layer_config = [
            random.choice((layer1, layer2))
            for layer1, layer2 in zip(genome1.layer_config, genome2.layer_config)
        ]

        # Ensure valid architecture: Conv layers before FC layers
        conv_layers = [layer for layer in self.layer_config if layer['layer_type'] == 'conv']
        fc_layers = [layer for layer in self.layer_config if layer['layer_type'] == 'fc']
        self.layer_config = conv_layers + fc_layers

    def mutate(self, config, generation=0):
        """
        Mutate the layer configurations with generation-aware scaling.
        """
        super().mutate(config)
        for layer in self.layer_config:
            if random.random() < 0.2:  # 20% chance to mutate each layer
                mutation_scale = max(1, 10 - generation // 5)  # Fine-tune scale reduction
                if layer['layer_type'] == 'conv':
                    layer['out_channels'] = max(16, layer['out_channels'] + random.choice([-mutation_scale, mutation_scale]))
                    layer['kernel_size'] = random.choice([3, 5])
                    layer['stride'] = max(1, layer['stride'] + random.choice([-1, 1]))
                    layer['padding'] = max(0, layer['padding'] + random.choice([-1, 1]))
                elif layer['layer_type'] == 'fc':
                    layer['fc_layer_size'] = max(32, layer['fc_layer_size'] + random.choice([-mutation_scale, mutation_scale]))

    def __str__(self):
        layer_info = "\n".join(
            f"Layer {i}: {layer}" for i, layer in enumerate(self.layer_config)
        )
        return f"{layer_info}\n{super().__str__()}"
