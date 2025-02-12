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
        mcts=None,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.strategy_placement = strategy_placement
        self.strategy_search = strategy_search

        self.config = config
        self.chromosome = chromosome
        self.range_evaluations = range_evaluations

        self.game_manager = game_manager
        self.mcts = mcts

    def simulate_game(self, game_manager, search_agent, placement_agent):
        """Simulate a Battleship game and return the move count."""

        current_state = game_manager.initial_state(placement_agent)

        while not game_manager.is_terminal(current_state):
            if self.mcts is not None:
                best_child = self.mcts.run(current_state, search_agent)
                move = best_child.move
            else:
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
            strategy="custom",
            chromosome=self.chromosome,
        )
        sum_move_count = 0

        for i in range(self.range_evaluations):
            move_count = self.simulate_game(game_manager, search_agent, placement_agent)
            sum_move_count += move_count

        # Return the genome fitness
        avg_moves = sum_move_count / self.range_evaluations
        return self.board_size**2 - avg_moves

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
    use_mcts=False,
    simulations_number=20,
    exploration_constant=1.41,
):
    # Searching Agent
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(10))

    game_manager = GameManager(size=board_size)
    mcts = None
    if use_mcts:
        mcts = MCTS(
            game_manager,
            simulations_number=simulations_number,
            exploration_constant=exploration_constant,
        )

    manager = NEAT_Manager(
        board_size,
        ship_sizes,
        strategy_placement,
        strategy_search,
        chromosome,
        range_evaluations,
        config,
        game_manager,
        mcts,
    )

    print("Config object as dictionary:")
    print(config.__dict__)

    winner = p.run(manager.eval_genomes, gen)

    # Print the winner's details
    print("\nBest genome details:")
    print(f"Genome ID: {winner.key}")
    print(f"Fitness: {winner.fitness}")
    print(f"Layer Configuration:")
    for layer in winner.layer_config:
        print(f"  {layer}")
    print("-" * 40)

    # Visualize the winner genome
    visualize.plot_stats(stats, ylog=False, view=True)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(
        CNNGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    run(
        config=config,
        gen=20,
        board_size=3,
        ship_sizes=[1, 2],
        strategy_placement="custom",
        strategy_search="neat",
        chromosome=[(0, 0, 0), (1, 1, 1)],
        range_evaluations=10,
        use_mcts=False,
        simulations_number=50,
        exploration_constant=1.41,
    )
