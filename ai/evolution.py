import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools

from game_logic.game_search_placing import Game
from game_logic.search_agent import SearchAgent
from game_logic.placement_agent import PlacementAgent

# Define the fitness and individual types for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualSearchAgent", SearchAgent, fitness=creator.FitnessMax)  # For SearchAgent
creator.create("IndividualPlacementAgent", PlacementAgent, fitness=creator.FitnessMax)  # For PlacementAgent

# Initialize the DEAP toolbox
toolbox = base.Toolbox()


# Helper functions to flatten and set neural network parameters
def flatten_params(network):
    """Flatten the network parameters into a single NumPy array."""
    return np.concatenate([param.data.numpy().flatten() for param in network.parameters()])


def set_params(network, flattened_params):
    """Set the network parameters from a flattened NumPy array."""
    offset = 0
    for param in network.parameters():
        num_params = param.numel()
        param.data.copy_(torch.tensor(flattened_params[offset:offset + num_params]).view(param.size()))
        offset += num_params


# Helper functions for the placing agent (chromosome)
def create_gene(ship_sizes):
    while True:  # Keep trying until all ships are placed correctly
        chromosome = []
        board = [[0] * 10 for _ in range(10)]  # Track occupied cells on the board
        valid = True

        for size in ship_sizes:
            placed = False
            for _ in range(100):  # Try up to 100 times to place the ship
                x = random.randint(0, 9)
                y = random.randint(0, 9)
                direction = random.randint(0, 1)  # 0 = horizontal, 1 = vertical

                if direction == 0 and y + size <= 10:  # Horizontal placement
                    if all(board[x][y + i] == 0 for i in range(size)):
                        for i in range(size):
                            board[x][y + i] = 1  # Mark the ship's position on the board
                        chromosome.append((x, y, direction))
                        placed = True
                        break

                elif direction == 1 and x + size <= 10:  # Vertical placement
                    if all(board[x + i][y] == 0 for i in range(size)):
                        for i in range(size):
                            board[x + i][y] = 1  # Mark the ship's position on the board
                        chromosome.append((x, y, direction))
                        placed = True
                        break

            if not placed:
                valid = False
                break  # Stop if a ship could not be placed

        if valid and len(chromosome) == len(ship_sizes):
            return chromosome  # Return the complete chromosome if all ships are placed correctly


def validate_placement(chromosome, ship_sizes):
    board = [[0] * 10 for _ in range(10)]  # Track occupied cells on the board

    for (x, y, direction), size in zip(chromosome, ship_sizes):
        if direction == 0 and y + size <= 10:  # Horizontal placement
            if all(board[x][y + i] == 0 for i in range(size)):
                for i in range(size):
                    board[x][y + i] = 1  # Mark the ship's position on the board
            else:
                return False

        elif direction == 1 and x + size <= 10:  # Vertical placement
            if all(board[x + i][y] == 0 for i in range(size)):
                for i in range(size):
                    board[x + i][y] = 1  # Mark the ship's position on the board
            else:
                return False
        else:
            return False

    return True


# Custom crossover and mutation functions for the placing agent
def custom_crossover_placement(ind1, ind2):
    """Custom crossover function for placing agents."""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]  # Swap ship placement between the two individuals
    return ind1, ind2


def custom_mutation_placement(individual, indpb):
    ship_sizes = [5, 4, 3, 3, 2]  # Ensure to use the same ship sizes for validation

    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutate the ship's position or direction randomly
            x, y, direction = individual[i]
            x = random.randint(0, 9)  # Mutate x-coordinate
            y = random.randint(0, 9)  # Mutate y-coordinate
            direction = random.randint(0, 1)  # Mutate direction
            individual[i] = (x, y, direction)

    # Validate the new placement; if invalid, revert the mutation
    if not validate_placement(individual, ship_sizes):
        individual = create_gene(ship_sizes)  # Reset to a valid placement if mutation results in invalid state

    return individual,


# Register custom crossover and mutation operators for placing agents
toolbox.register("mate_chromosome", custom_crossover_placement)
toolbox.register("mutate_chromosome", custom_mutation_placement, indpb=0.2)

# Register selection operator
toolbox.register("select", tools.selTournament, tournsize=3)


# Custom crossover and mutation for SearchAgent (NN)
def search_agent_crossover(agent1, agent2):
    """Custom crossover logic for SearchAgent that works with NumPy arrays."""
    for i in range(len(agent1.strategy_params)):
        if random.random() < 0.5:
            agent1.strategy_params[i], agent2.strategy_params[i] = agent2.strategy_params[i], agent1.strategy_params[i]
    return agent1, agent2


def search_agent_mutation(agent, indpb=0.2):
    """Custom mutation logic for SearchAgent using NumPy arrays."""
    for i in range(len(agent.strategy_params)):
        if random.random() < indpb:
            agent.strategy_params[i] += np.random.normal(0, 0.1)  # Add Gaussian noise
    return agent,


# Register custom crossover and mutation operators for SearchAgent
toolbox.register("mate_search_agent", search_agent_crossover)
toolbox.register("mutate_search_agent", search_agent_mutation)


class Evolution:
    def __init__(self, board_size, ship_sizes):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        toolbox.register("evaluate_population", self.evaluate_population)

    def simulate_game(self, placing_agent, search_agent):
        """Simulate a Battleship game and return the move count."""
        game = Game(board_size=self.board_size, sizes=self.ship_sizes, placing=placing_agent, search=search_agent)
        game.placing.show_ships()
        while not game.game_over:
            game.play_turn()
        return game.move_count

    def evaluate(self, placing_agent, search_agent):
        # Only set parameters if they have been updated
        if search_agent.strategy_params is not None:
            set_params(search_agent.strategy, search_agent.strategy_params)

        # Run the game simulation to get the move count
        move_count = self.simulate_game(placing_agent, search_agent)

        return move_count

    def evaluate_population(self, population_placing, population_search):
        # Initialize total fitness accumulators for placing agents
        placing_fitness_accumulators = {agent: 0.0 for agent in population_placing}

        for search_agent in population_search:
            total_search_fitness = 0

            # Evaluate the search agent against every placing agent (round-robin)
            for placing_agent in population_placing:
                fitness = self.evaluate(placing_agent, search_agent)

                # Reset the search agent for the next game
                search_agent.reset()

                # Accumulate fitness scores
                total_search_fitness += fitness
                placing_fitness_accumulators[placing_agent] += fitness

            # Set the fitness for the search agent (lower move count = better fitness)
            search_agent.fitness.values = (-total_search_fitness / len(population_placing),)

        # Set the fitness for placing agents (higher move count = better fitness)
        for placing_agent, total_fitness in placing_fitness_accumulators.items():
            placing_agent.fitness.values = (total_fitness / len(population_search),)

    def evolve(self):
        pop_placing_agents = self.initialize_placing_population(n=15)
        pop_search_agents = self.initialize_search_population(n=15)

        avg_moves_over_gens = []

        for gen in range(20):
            print(f"Generation {gen}")

            # Evaluate the population
            toolbox.evaluate_population(pop_placing_agents, pop_search_agents)

            # Calculate average moves for this generation (use the placing agents' fitness for direct move counts)
            avg_moves = sum(agent.fitness.values[0] for agent in pop_placing_agents) / len(pop_placing_agents)
            avg_moves_over_gens.append(avg_moves)

            print(f"Avg Moves: {avg_moves}")

            # Selection, crossover, and mutation steps...
            offspring_placing = toolbox.select(pop_placing_agents, len(pop_placing_agents))
            offspring_search = toolbox.select(pop_search_agents, len(pop_search_agents))

            # Apply custom crossover and mutation for placing agents
            for i in range(1, len(offspring_placing), 2):
                offspring_placing[i - 1].strategy.chromosome, offspring_placing[
                    i].strategy.chromosome = toolbox.mate_chromosome(
                    offspring_placing[i - 1].strategy.chromosome, offspring_placing[i].strategy.chromosome
                )
            for i in range(len(offspring_placing)):
                offspring_placing[i].strategy.chromosome = \
                toolbox.mutate_chromosome(offspring_placing[i].strategy.chromosome)[0]

            offspring_placing = [creator.IndividualPlacementAgent(self.board_size, self.ship_sizes, strategy="custom",
                                                                  chromosome=ind.strategy.chromosome)
                                 for ind in offspring_placing]

            # Apply custom crossover and mutation for search agents
            for i in range(1, len(offspring_search), 2):
                toolbox.mate_search_agent(offspring_search[i - 1], offspring_search[i])
            for i in range(len(offspring_search)):
                toolbox.mutate_search_agent(offspring_search[i])

            offspring_search = [creator.IndividualSearchAgent(self.board_size, self.ship_sizes, strategy="nn_search")
                                for ind in offspring_search]
            for agent in offspring_search:
                agent.strategy_params = flatten_params(agent.strategy).copy()

            # Replace the current population with the offspring
            pop_placing_agents[:] = offspring_placing
            pop_search_agents[:] = offspring_search

        # Plot the metrics after evolution
        self.plot_metrics(avg_moves_over_gens)

    def initialize_search_population(self, n):
        """Initialize the search agent population."""
        population = []
        for _ in range(n):
            search_agent = creator.IndividualSearchAgent(self.board_size, self.ship_sizes, strategy="nn_search")
            # Extract parameters once and store them as a NumPy array
            search_agent.strategy_params = flatten_params(search_agent.strategy).copy()
            population.append(search_agent)
        return population

    def initialize_placing_population(self, n):
        """Initialize the placing agent population."""
        population = []
        for _ in range(n):
            placing_agent = creator.IndividualPlacementAgent(self.board_size, self.ship_sizes, strategy="custom",
                                                             chromosome=create_gene(self.ship_sizes))
            population.append(placing_agent)
        return population

    def plot_metrics(self, avg_moves_over_gens):
        plt.plot(avg_moves_over_gens, label='Avg Moves')
        plt.xlabel('Generation')
        plt.ylabel('Number of Moves')
        plt.title('Agent Performance Over Generations')
        plt.legend()
        plt.show()

    def save_best_solution(self, best_placing, best_search):
        """Save the best chromosome and NN parameters."""
        with open("best_chromosome_placing.txt", "w") as f:
            f.write(str(best_placing.strategy.chromosome))
        torch.save(best_search.strategy.state_dict(), "best_nn_search.pth")


if __name__ == "__main__":
    board_size = 10
    ship_sizes = [5, 4, 3, 3, 2]
    environment = Evolution(board_size=board_size, ship_sizes=ship_sizes)
    environment.evolve()
