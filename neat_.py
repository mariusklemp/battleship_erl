import os
import neat

from game_logic.game_search_placing import Game
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent


class NEAT_Manager:
    def __init__(self, board_size, ship_sizes, config):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.config = config

    def simulate_game(self, net):
        """Simulate a Battleship game and return the move count."""

        search_agent = SearchAgent(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy="neat",
            net=net,
        )

        game = Game(
            placing=PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="random"),
            search=search_agent,
        )

        while not game.game_over:
            game.play_turn()
        return game.move_count

    def evaluate(self, genome):
        """Evaluate the genome fitness."""
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        sum_move_count = 0
        for i in range(5):
            # Play the search against a random placing agent 5 times
            move_count = self.simulate_game(net)
            sum_move_count += move_count

        # Update the genome fitness
        print("Average movecount: ", sum_move_count/5)
        genome.fitness += sum_move_count/5



def eval_genomes(genomes, config):
    """Evaluate the fitness of each genome in the population."""
    manager = NEAT_Manager(board_size=10, ship_sizes=[5, 4, 3, 3, 2], config=config)
    for i, (genome_id, genome) in enumerate(genomes):
        print(f"Genome {i} with ID: {genome_id}")
        if i == len(genomes) - 1:
            break
        genome.fitness = 0
        manager.evaluate(genome)


def run(config):
    # Searching Agent
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)
    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    board_size = 10
    ship_sizes = [5, 4, 3, 3, 2]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run(config)
