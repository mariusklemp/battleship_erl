import torch
import random

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


# Helper functions for the placing agent (chromosome)
def create_gene(ship_sizes):
    chromosone = []
    board = [[0] * 10 for _ in range(10)]  # Track occupied cells on the board

    for size in ship_sizes:
        valid = False
        while not valid:
            x = random.randint(0, 9)  # X-coordinate (0-9 for a 10x10 board)
            y = random.randint(0, 9)  # Y-coordinate (0-9 for a 10x10 board)
            direction = random.randint(0, 1)  # 0 = horizontal, 1 = vertical

            # Check if the ship fits on the board and doesn't overlap
            if direction == 0 and y + size <= 10:  # Horizontal placement
                if all(board[x][y + i] == 0 for i in range(size)):
                    for i in range(size):
                        board[x][y + i] = 1  # Mark the ship's position on the board
                    chromosone.append((x, y, direction))
                    valid = True

            elif direction == 1 and x + size <= 10:  # Vertical placement
                if all(board[x + i][y] == 0 for i in range(size)):
                    for i in range(size):
                        board[x + i][y] = 1  # Mark the ship's position on the board
                    chromosone.append((x, y, direction))
                    valid = True

    return chromosone


# TODO fix custom crossover and mutation functions. We have to validate the placement of ships
def custom_crossover_placement(ind1, ind2):
    """Custom crossover function for placing agents."""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]  # Swap ship placement between the two individuals
    return ind1, ind2


def custom_mutation_placement(individual, indpb):
    """Custom mutation function for placing agents."""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutate the ship's position or direction randomly
            x, y, direction = individual[i]
            x = random.randint(0, 9)  # Mutate x-coordinate
            y = random.randint(0, 9)  # Mutate y-coordinate

            direction = random.randint(0, 1)  # Mutate direction
            individual[i] = (x, y, direction)
    return individual,


# Register custom crossover and mutation operators for placing agents
toolbox.register("mate_chromosome", custom_crossover_placement)
toolbox.register("mutate_chromosome", custom_mutation_placement, indpb=0.2)

# Register selection operator
toolbox.register("select", tools.selTournament, tournsize=3)


# Custom crossover and mutation for SearchAgent (NN)
def search_agent_crossover(agent1, agent2):
    """Custom crossover logic for SearchAgent that blends neural network weights."""

    # Perform a weight crossover (blending)
    for param1, param2 in zip(agent1.strategy.parameters(), agent2.strategy.parameters()):
        alpha = random.random()
        param1.data.copy_(alpha * param1.data + (1 - alpha) * param2.data)

    return agent1, agent2


def search_agent_mutation(agent):
    """Custom mutation logic for SearchAgent by adding Gaussian noise to weights."""

    # Apply Gaussian mutation to the weights of the neural network
    for param in agent.strategy.parameters():
        noise = torch.normal(mean=0, std=0.1, size=param.data.size())
        param.data.add_(noise)
    return agent,


# Register custom crossover and mutation operators for SearchAgent
toolbox.register("mate_search_agent", search_agent_crossover)
toolbox.register("mutate_search_agent", search_agent_mutation)


class Evolution:
    def __init__(self, board_size, ship_sizes):
        self.board_size = board_size
        self.ship_sizes = ship_sizes

        # Register the evaluation function in the toolbox
        toolbox.register("evaluate_population", self.evaluate_population)

    def simulate_game(self, placing_agent, search_agent):
        """Simulate a Battleship game and return the move count."""
        game = Game(board_size=self.board_size, sizes=self.ship_sizes, placing=placing_agent, search=search_agent)

        while not game.game_over:
            game.play_turn()

        print(f"Result: Used {game.move_count} moves!")
        # Return the number of moves it took to find all ships
        return game.move_count

    def evaluate(self, placing_agent, search_agent):
        # Simulate a game and get the move count
        move_count = self.simulate_game(placing_agent, search_agent)

        # Fitness assignment
        placing_agent_fitness = move_count  # More moves means better placing (avoiding detection)
        search_agent_fitness = -move_count  # Fewer moves means better search (finding quickly)

        return placing_agent_fitness, search_agent_fitness

    def evaluate_population(self, population_placing, population_search):
        for search_agent in population_search:
            total_search_fitness = 0
            total_placing_fitness = 0

            # Play against 5 randomly sampled placing agents
            for _ in range(3):
                placing_agent = random.choice(population_placing)
                placing_fitness, search_fitness = self.evaluate(placing_agent, search_agent)
                search_agent.reset()
                total_search_fitness += search_fitness
                total_placing_fitness += placing_fitness

            # Set the search agent fitness (average across the games)
            search_agent.fitness.values = (total_search_fitness / 3,)  # 5 random + 1 from Hall of Fame
            placing_agent.fitness.values = (total_placing_fitness / 3,)  # Same for placing agent

    # TODO implement HoF (Hall of Fame) to store the best individuals after each generation
    def evolve(self):
        pop_placing_agents = self.initialize_placing_population(n=50)
        pop_search_agents = self.initialize_search_population(n=50)

        for gen in range(10):
            print(f"Generation {gen}")

            # Evaluate the population
            toolbox.evaluate_population(pop_placing_agents, pop_search_agents)

            # Select the next generation individuals
            offspring_placing = toolbox.select(pop_placing_agents, len(pop_placing_agents))
            offspring_search = toolbox.select(pop_search_agents, len(pop_search_agents))

            # Apply custom crossover and mutation for placing agents
            for i in range(1, len(offspring_placing), 2):
                # Apply crossover to the chromosomes of two agents
                offspring_placing[i - 1].strategy.chromosome, offspring_placing[
                    i].strategy.chromosome = toolbox.mate_chromosome(
                    offspring_placing[i - 1].strategy.chromosome, offspring_placing[i].strategy.chromosome
                )

            # Apply mutation for each placing agent's chromosome
            for i in range(len(offspring_placing)):
                offspring_placing[i].strategy.chromosome = toolbox.mutate_chromosome(
                    offspring_placing[i].strategy.chromosome
                )[0]

            # Create new instances of PlacementAgent using updated chromosomes
            offspring_placing = [creator.IndividualPlacementAgent(self.board_size, self.ship_sizes,
                                                                  strategy="custom", chromosome=ind.strategy.chromosome)
                                 for ind in offspring_placing]

            # Apply custom crossover and mutation for search agents
            for i in range(1, len(offspring_search), 2):
                # Crossover neural networks
                toolbox.mate_search_agent(offspring_search[i - 1], offspring_search[i])
            for i in range(len(offspring_search)):
                # Mutate neural networks
                toolbox.mutate_search_agent(offspring_search[i])

            # Create new instances of SearchAgent using updated neural networks
            offspring_search = [
                creator.IndividualSearchAgent(self.board_size, self.ship_sizes, strategy="nn_search",
                                              weights=ind.strategy.state_dict())
                for ind in offspring_search
            ]

            # Replace the current population with the offspring
            pop_placing_agents[:] = offspring_placing
            pop_search_agents[:] = offspring_search

        # Store the best solutions
        best_placing = tools.selBest(pop_placing_agents, 1)[0]
        best_search = tools.selBest(pop_search_agents, 1)[0]

        # Save the best chromosome and neural network
        self.save_best_solution(best_placing, best_search)

    def initialize_search_population(self, n):
        """Initialize the search agent population."""
        population = []
        for _ in range(n):
            search_agent = creator.IndividualSearchAgent(self.board_size, self.ship_sizes, strategy="nn_search")
            population.append(search_agent)
        return population

    def initialize_placing_population(self, n):
        """Initialize the placing agent population."""
        population = []
        for _ in range(n):
            placing_agent = creator.IndividualPlacementAgent(self.board_size, self.ship_sizes, strategy="custom",
                                                             chromosome=create_gene(ship_sizes))
            population.append(placing_agent)
        return population

    def save_best_solution(self, best_placing, best_search):
        """Save the best chromosome and NN parameters."""
        # Save chromosome for placing agent
        with open("best_chromosome_placing.txt", "w") as f:
            f.write(str(best_placing.strategy.chromosome))

        # Save the neural network parameters (best_search.strategy is the NN)
        torch.save(best_search.strategy.state_dict(), "best_nn_search.pth")
        print("Search agent NN parameters saved")


if __name__ == "__main__":
    # Define the board size and ship sizes
    board_size = 10
    ship_sizes = [5, 4, 3, 3, 2]

    # Run the evolution process
    environment = Evolution(board_size=board_size, ship_sizes=ship_sizes)
    environment.evolve()
