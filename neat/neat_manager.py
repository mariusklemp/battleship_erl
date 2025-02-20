import os
import neat
import tqdm
from tqdm import tqdm

from CNN_genome import CNNGenome
from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from ai.mcts import MCTS
import visualize
from convolutional_neural_network import ConvolutionalNeuralNetwork


class NEAT_Manager:
    def __init__(
            self,
            board_size,
            ship_sizes,
            strategy_placement,
            strategy_search,
            chromosome,
            range_evaluations,
            config,
            game_manager,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.strategy_placement = strategy_placement
        self.strategy_search = strategy_search

        self.config = config
        self.chromosome = chromosome
        self.range_evaluations = range_evaluations

        self.game_manager = game_manager

    def simulate_game(self, game_manager, search_agent, placement_agent):
        """Simulate a Battleship game and return the move count."""

        current_state = game_manager.initial_state(placement_agent)

        while not game_manager.is_terminal(current_state):
            move = search_agent.strategy.find_move(current_state)

            current_state = game_manager.next_state(
                current_state, move, current_state.placing
            )

        return current_state.move_count

    def evaluate(self, game_manager, net):
        """Simulate games to evaluate the genome fitness."""

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy=self.strategy_search,
            net=net,
        )
        placement_agent = PlacementAgent(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy="chromosome",
            chromosome=self.chromosome,
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
            net = ConvolutionalNeuralNetwork.create(genome=genome, config=config)
            genome.fitness = self.evaluate(self.game_manager, net)


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
    # p.add_reporter(neat.Checkpointer(10))

    game_manager = GameManager(size=board_size)

    manager = NEAT_Manager(
        board_size,
        ship_sizes,
        strategy_placement,
        strategy_search,
        chromosome,
        range_evaluations,
        config,
        game_manager,
    )

    print("Config object as dictionary:")
    print(config.__dict__)

    winner = p.run(manager.eval_genomes, gen)

    # Visualize the winner genome
    visualize.plot_stats(statistics=stats, best_possible=(board_size ** 2 - sum(ship_sizes)), ylog=False, view=True)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    # === Static Parameters ===
    BOARD_SIZE = 3
    SHIP_SIZES = [2]
    NUM_GENERATIONS = 10
    RANGE_EVALUATIONS = 5
    MUTPB = 0.2
    # =======================================

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
        strategy_placement="custom",
        strategy_search="neat",
        chromosome=[(0, 0, 0)],
        range_evaluations=RANGE_EVALUATIONS,
    )
