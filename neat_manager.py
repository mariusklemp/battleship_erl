import os
import neat
import tqdm
from tqdm import tqdm

from CNN_genome import CNNGenome
from game_logic.game_search_placing import GameManager
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
        config,
        game_manager,
        mcts=None,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.config = config
        self.strategy_placement = strategy_placement
        self.strategy_search = strategy_search

        self.chromosome = chromosome
        self.game_manager = game_manager
        self.mcts = mcts

    def simulate_game(self, game_manager, net):
        """Simulate a Battleship game and return the move count."""
        search_agent = SearchAgent(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy=self.strategy_search,
            net=net,
        )

        placement_agent = PlacementAgent(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy="random",
        )

        current_state = game_manager.initial_state(placement_agent)

        placement_agent.show_ships()

        while not game_manager.is_terminal(current_state):
            game_manager.show_board(current_state)
            if self.mcts is not None:
                best_child = self.mcts.run(current_state, search_agent)
                move = best_child.move
                # self.mcts.print_tree()
                if move is None:
                    print("MCTS returned None move!")
                    break
            else:
                move = search_agent.strategy.find_move(current_state)

            current_state = game_manager.next_state(
                current_state, move, game_manager.placing
            )

        return current_state.move_count

    def evaluate(self, game_manager, genome, net):
        """Evaluate the genome fitness."""
        sum_move_count = 0
        range_count = 5
        for i in tqdm(range(range_count), desc="Generation simulations", leave=False):
            move_count = self.simulate_game(game_manager, net)
            sum_move_count += move_count

        # Update the genome fitness
        avg_moves = sum_move_count / range_count
        genome.fitness += max(0, (self.board_size**2 - avg_moves))

    def eval_genomes(self, genomes, config):
        """Evaluate the fitness of each genome in the population."""

        for i, (genome_id, genome) in enumerate(
            tqdm(genomes, desc="Evaluating generation")
        ):
            genome.fitness = 0
            net = ConvolutionalNeuralNetwork.create(genome=genome, config=config)
            self.evaluate(self.game_manager, genome, net)
            # print("Fitness: ", genome.fitness)


def run(
    config,
    gen,
    board_size,
    ship_sizes,
    strategy_placement,
    strategy_search,
    chromosome,
    use_mcts=False,
    simulations_number=50,
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
        config,
        game_manager,
        mcts,
    )

    print("Config object as dictionary:")
    print(config.__dict__)

    # Print the first 50 individuals in the population
    for i, genome in enumerate(p.population.values()):
        print(f"Genome ID: {genome.key}, Fitness: {genome.fitness}")
        print(f"Layer Configuration:")
        for layer in genome.layer_config:
            print(f"  {layer}")
        print("-" * 40)  # Separator between genomes

    winner = p.run(manager.eval_genomes, gen)

    # Print the winner's details
    print("\nBest genome details:")
    print(f"Genome ID: {winner.key}")
    print(f"Fitness: {winner.fitness}")
    print(f"Layer Configuration:")
    for layer in winner.layer_config:
        print(f"  {layer}")
    print("-" * 40)  # Separator for clarity

    # Visualize the winner genome
    visualize.plot_stats(stats, ylog=False, view=True)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(
        CNNGenome,  # Use custom genome class
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    run(
        config=config,
        gen=10,
        board_size=5,
        ship_sizes=[1, 2, 4],
        strategy_placement="custom",
        strategy_search="neat",
        chromosome=[(0, 0, 0), (1, 2, 1), (4, 0, 1)],
        use_mcts=True,
        simulations_number=50,
        exploration_constant=1.41,
    )
