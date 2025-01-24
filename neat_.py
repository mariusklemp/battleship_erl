import os
import neat

from customGenomes import CNNGenome
from game_logic.game_search_placing import Game
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
import visualize
from strategies.search.Deep_NEAT import DeepNEATCNN


class NEAT_Manager:
    def __init__(
        self,
        board_size,
        ship_sizes,
        strategy_placement,
        strategy_search,
        chromosome,
        config,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.config = config
        self.strategy_placement = strategy_placement
        self.strategy_search = strategy_search

        self.chromosome = chromosome

    def simulate_game(self, net):
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
            strategy=self.strategy_placement,
            chromosome=self.chromosome,
        )

        game = Game(
            placing=placement_agent,
            search=search_agent,
        )

        # game.placing.show_ships()

        while not game.game_over:
            game.play_turn()
        return game.move_count

    def evaluate(self, genome, net):
        """Evaluate the genome fitness."""
        sum_move_count = 0
        range_count = 5
        for i in range(range_count):
            # Play the search against a random placing agent 5 times
            move_count = self.simulate_game(net)
            sum_move_count += move_count

        # Update the genome fitness
        avg_moves = sum_move_count / range_count
        genome.fitness += self.board_size**2 - avg_moves
        # print("Average movecount: ", avg_moves)

    def eval_genomes(self, genomes, config):
        """Evaluate the fitness of each genome in the population."""
        for i, (genome_id, genome) in enumerate(genomes):
            # print(f"Genome {i} with ID: {genome_id}")
            genome.fitness = 0
            # net = neat.nn.FeedForwardNetwork.create(genome, config)
            net = DeepNEATCNN(
                genome=genome, board_size=self.board_size, config=config
            )  # Creates the CNN instance
            """
            For å kjøre CNN endre find_move i NEAT_search.py
            """
            self.evaluate(genome, net)
            # print("Fitness: ", genome.fitness)


def run(
    config, gen, board_size, ship_sizes, strategy_placement, strategy_search, chromosome
):
    # Searching Agent
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(10))

    manager = NEAT_Manager(
        board_size, ship_sizes, strategy_placement, strategy_search, chromosome, config
    )

    winner = p.run(manager.eval_genomes, gen)
    print("Run completed")
    print("\nBest genome details:")
    print("Fitness: {}".format(winner.fitness))

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
        gen=20,
        board_size=5,
        ship_sizes=[1, 2, 1],
        strategy_placement="custom",
        strategy_search="neat",
        chromosome=[(0, 0, 0), (2, 1, 1), (4, 0, 1)],
    )
