import os
import neat

from game_logic.game_manager import Game
from game_logic.search_agent import SearchAgent


class NEAT_Manager:
    def __init__(self, board_size, ship_sizes, config):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.config = config

        self.placing_agents = []

    def simulate_game(self, placing_agent, genome):
        """Simulate a Battleship game and return the move count."""
        net1 = neat.nn.FeedForwardNetwork.create(genome, self.config)
        search_agent = SearchAgent(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy="neat",
            net=net1,
        )

        game = Game(
            board_size=self.board_size,
            sizes=self.ship_sizes,
            placing=placing_agent,
            search=search_agent,
        )
        game.placing.show_ships()
        while not game.game_over:
            game.play_turn()
        return game.move_count

    def evaluate(self, genome):
        """Evaluate the genome fitness."""
        for placing_agent in self.placing_agents:
            # Run the game simulation to get the move count
            move_count = self.simulate_game(placing_agent, genome, self.config)

            # Update the genome fitness
            genome.fitness += move_count


def eval_genomes(genomes, config):
    """Evaluate the fitness of each genome in the population."""
    manager = NEAT_Manager(board_size=10, ship_sizes=[5, 4, 3, 3, 2], config=config)
    for i, (genome_id, genome) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome.fitness = 0
        manager.evaluate(genome)


def run_neat(config):
    # Searching Agent
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)


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
    run_neat(config)
