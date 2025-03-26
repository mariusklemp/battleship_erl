import configparser
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import neat
import tqdm
from tqdm import tqdm

from neat_system.cnn_genome import CNNGenome
from game_logic.game_manager import GameManager
from game_logic.search_agent import SearchAgent
import visualize
from ai.model import ANET
from game_logic.placement_agent import PlacementAgent
from neat_system.weight_reporter import WeightStatsReporter

from neat_system.convolutional_neural_network import ConvolutionalNeuralNetwork



class NEAT_Manager:
    def __init__(
            self,
            board_size,
            ship_sizes,
            strategy_placement,
            strategy_search,
            range_evaluations,
            config,
            game_manager,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.strategy_placement = strategy_placement
        self.strategy_search = strategy_search

        self.config = config
        self.range_evaluations = range_evaluations

        self.game_manager = game_manager

    def set_placement_agents(self, placement_agents):
        self.placement_agents = placement_agents

    def simulate_game(self, game_manager, search_agent, placement_agent):
        """Simulate a Battleship game and return the move count."""

        current_state = game_manager.initial_state(placement_agent)

        while not game_manager.is_terminal(current_state):
            move = search_agent.strategy.find_move(current_state)

            current_state = game_manager.next_state(
                current_state, move, current_state.placing
            )

        return current_state.move_count

    def evaluate(self, game_manager, net, placement_agent):
        """Simulate games to evaluate the genome fitness."""

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy=self.strategy_search,
            net=net,
        )

        sum_move_count = 0

        for i in range(self.range_evaluations):
            move_count = self.simulate_game(game_manager, search_agent, placement_agent)
            sum_move_count += move_count

        # Return the genome fitness
        avg_moves = sum_move_count / self.range_evaluations
        return self.board_size ** 2 - avg_moves

    def eval_genomes(self, genomes, config):
        """Evaluate the fitness of each genome in the population."""

        for i, (genome_id, genome) in enumerate(
                tqdm(genomes, desc="Evaluating generation")
        ):
            for placement_agent in self.placement_agents:
                net = ANET(genome=genome, config=config) # New

                # net = ConvolutionalNeuralNetwork.create(genome=genome, config=config)

                genome.fitness = self.evaluate(self.game_manager, net, placement_agent)


def run(
        config,
        gen,
        board_size,
        ship_sizes,
        strategy_placement,
        strategy_search,
        chromosome,
        range_evaluations,
):
    # Searching Agent
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Attach custom WeightStatsReporter
    weight_stats_reporter = WeightStatsReporter()
    p.add_reporter(weight_stats_reporter)

    # p.add_reporter(neat.Checkpointer(10))

    game_manager = GameManager(size=board_size)

    manager = NEAT_Manager(
        board_size,
        ship_sizes,
        strategy_placement,
        strategy_search,
        range_evaluations,
        config,
        game_manager,
    )
    placement_agents = []
    placement_agents.append(
        PlacementAgent(board_size, ship_sizes, strategy_placement, chromosome)
    )
    manager.set_placement_agents(placement_agents)

    p.run(manager.eval_genomes, gen)

    visualize.visualize_hof(statistics=stats)
    visualize.plot_weight_stats(weight_stats_reporter.get_weight_stats())
    visualize.plot_species_weight_stats(
        weight_stats_reporter.get_species_weight_stats()
    )
    species_analysis = visualize.analyze_species_from_population(p.species)
    visualize.plot_species_analysis(species_analysis)
    visualize.visualize_species(stats)
    visualize.plot_stats(
        statistics=stats,
        best_possible=(board_size ** 2 - sum(ship_sizes)),
        ylog=False,
        view=True,
    )
    visualize.plot_fitness_boxplot(stats)
    from neat_system.cnn_layers import global_innovation_registry

    visualize.plot_innovation_registry(global_innovation_registry)

    best_genomes = stats.best_genomes(5)
    genomes = []
    for genome in best_genomes:
        genomes.append((genome, ""))
    visualize.plot_multiple_genomes(genomes, "Best Genomes")


def get_kernel_sizes(board_size):
    # Use the board size if it's odd; otherwise, subtract 1 to get the largest odd
    max_odd = board_size if board_size % 2 == 1 else board_size - 1
    # Calculate a middle kernel size that's also odd
    mid = (1 + max_odd) // 2
    if mid % 2 == 0:
        mid += 1
    # Remove duplicates (e.g., for board_size=3, 1 may appear twice)
    return list(dict.fromkeys([1, mid, max_odd]))


def get_strides(board_size):
    # For strides, we'll keep it simple: always include 1 and board_size//2
    return list(dict.fromkeys([1, board_size // 2]))


def get_paddings(board_size):
    # For paddings, use 0 and board_size//4
    return list(dict.fromkeys([0, board_size // 4]))


def get_pool_sizes(board_size):
    """Returns a list of valid pooling sizes, ensuring they fit within the board."""
    max_pool = (
        board_size if board_size % 2 == 0 else board_size - 1
    )  # Largest even size
    mid_pool = max(2, board_size // 2)  # Half the board, but at least 2
    return list(dict.fromkeys([2, mid_pool, max_pool]))


def get_pool_strides(board_size):
    """Returns a list of valid pooling strides, ensuring valid downsampling."""
    return list(dict.fromkeys([1, board_size // 2]))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    # === Static Parameters ===
    BOARD_SIZE = 5
    POPULATION_SIZE = 50
    SHIP_SIZES = [3, 2, 2]
    CHROMOSOME = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
    ]
    NUM_GENERATIONS = 5
    RANGE_EVALUATIONS = 5
    MUTATE_ARCHITECTURE = True
    CROSSOVER_ARCHITECTURE = True
    MUTATE_WEIGHTS = True
    CROSSOVER_WEIGHTS = False
    # =======================================

    # Read the existing config file
    cp = configparser.ConfigParser()
    cp.read(config_path)

    # Update [NEAT] section values
    cp["NEAT"]["fitness_threshold"] = str(BOARD_SIZE ** 2 - sum(SHIP_SIZES))
    cp["NEAT"]["pop_size"] = str(POPULATION_SIZE)

    # Update [CNNGenome] section values
    cp["CNNGenome"]["input_size"] = str(BOARD_SIZE)
    cp["CNNGenome"]["output_size"] = str(BOARD_SIZE ** 2)
    cp["CNNGenome"]["kernel_sizes"] = str(get_kernel_sizes(BOARD_SIZE))
    cp["CNNGenome"]["strides"] = str(get_strides(BOARD_SIZE))
    cp["CNNGenome"]["paddings"] = str(get_paddings(BOARD_SIZE))

    cp["CNNGenome"]["pool_sizes"] = str(get_pool_sizes(BOARD_SIZE))
    cp["CNNGenome"]["pool_strides"] = str(get_pool_strides(BOARD_SIZE))

    cp["CNNGenome"]["mutate_architecture"] = str(MUTATE_ARCHITECTURE)
    cp["CNNGenome"]["mutate_weights"] = str(MUTATE_WEIGHTS)
    cp["CNNGenome"]["crossover_architecture"] = str(CROSSOVER_ARCHITECTURE)
    cp["CNNGenome"]["crossover_weights"] = str(CROSSOVER_WEIGHTS)

    # Write the updated config back to disk
    with open(config_path, "w") as f:
        cp.write(f)

    # Now load the NEAT configuration from the updated config file
    config = neat.Config(
        CNNGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    run(
        config=config,
        gen=NUM_GENERATIONS,
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        strategy_placement="random",
        strategy_search="nn_search",
        chromosome=CHROMOSOME,
        range_evaluations=RANGE_EVALUATIONS,
    )
