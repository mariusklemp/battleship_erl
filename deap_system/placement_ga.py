import gc
import random
from functools import partial
import sys
import os

from game_logic.game_manager import GameManager
from game_logic.search_agent import SearchAgent

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import base, creator
import copy
from deap import tools

from game_logic.placement_agent import PlacementAgent

from deap_system.helpers import (
    is_gene_valid,
    mark_board,
    random_valid_gene,
    local_mutation_gene,
)

# ---------------------------------------------------------------------
# Define DEAP Fitness and Individual Types
# ---------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPlacementAgent", PlacementAgent, fitness=creator.FitnessMax)

# Global DEAP toolbox
toolbox = base.Toolbox()


class ChromosomeHallOfFame:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.items = []

    def update(self, population):
        best = tools.selBest(population, self.maxsize)
        self.items = [copy.deepcopy(agent.strategy.chromosome) for agent in best]


class ChromosomeHallOfShame:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.items = []

    def update(self, population):
        worst = tools.selWorst(population, self.maxsize)
        self.items = [copy.deepcopy(agent.strategy.chromosome) for agent in worst]


class PlacementGeneticAlgorithm:
    def __init__(
            self,
            game_manager,
            board_size,
            ship_sizes,
            population_size,
            num_generations,
            elite_size=1,
            MUTPB=0.2,
            TOURNAMENT_SIZE=3,
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_size = elite_size
        toolbox.register("evaluate_population", self.evaluate_population)

        # Metrics for visualization:
        self.avg_moves_over_gens = []
        self.boards_over_generations = []  # Average board overlay per generation
        # Use the custom hall-of-fame and hall-of-shame classes.
        self.hof = ChromosomeHallOfFame(maxsize=1)
        self.hos = ChromosomeHallOfShame(maxsize=1)
        # Record halls per generation
        self.hof_per_generation = []
        self.hos_per_generation = []

        self.avg_fitness_over_gens = []
        self.diversity_over_gens = []
        self.sparsity_over_gens = []
        self.percent_vertical_over_gens = []  # Only store vertical percentage.

        # Set up game environment.
        self.game_manager = game_manager

        # Initialize population.
        self.pop_placing_agents = []
        self.population_chromosomes = []  # Store chromosomes separately.

        # Register DEAP operators with our custom methods.
        toolbox.register("mate_chromosome", self.custom_crossover_placement)
        toolbox.register(
            "mutate_chromosome",
            partial(self.custom_mutation_placement, indpb=MUTPB),
        )
        toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # ------------------- Game Simulation and Evaluation -------------------
    def simulate_game(self, game_manager, placing_agent, search_agent):
        """Simulate a Battleship game and return the move count, hits, and misses."""
        current_state = game_manager.initial_state(placing=placing_agent)
        # Count total hits and misses at the end instead of tracking during the game
        while not game_manager.is_terminal(current_state):
            result = search_agent.strategy.find_move(current_state)
            move, distribution = (result, None) if not isinstance(result, tuple) else result

            current_state = game_manager.next_state(current_state, move)

        hits = sum(square == 1 for square in current_state.board[1])
        misses = sum(square == 1 for square in current_state.board[2])

        return current_state.move_count, hits, misses

    def evaluate_population(
            self, population_placing, game_manager, search_agents
    ):
        """
        Evaluate each placing agent by simulating games against a pool of opponents
        and assigning the average performance as fitness.
        """

        for placing_agent in population_placing:
            fitness_scores = []
            for search_agent in search_agents:
                moves, hits, misses = self.simulate_game(
                    game_manager, placing_agent, search_agent
                )
                fitness_scores.append(moves)

            placing_agent.fitness.values = (sum(fitness_scores) / len(fitness_scores),)

    def initialize_placing_population(self, n):
        population = []
        self.population_chromosomes = []  # separate list for chromosomes
        for i in range(n):
            placing_agent = creator.IndividualPlacementAgent(
                self.board_size, self.ship_sizes, strategy="chromosome", name=f"placing_{i}"
            )
            population.append(placing_agent)
            # Store a deep copy of the initial chromosome
            self.population_chromosomes.append(copy.deepcopy(placing_agent.strategy.chromosome))
        self.pop_placing_agents = population


    # ------------------- Evolutionary Process -------------------
    def trigger_evaluate_population(self, search_agents):
        # This will update fitness for each placing agent.
        toolbox.evaluate_population(self.pop_placing_agents, self.game_manager, search_agents)

    def evolve(self):
        # Update Hall of Fame and Hall of Shame.
        self.hof.update(self.pop_placing_agents)
        self.hos.update(self.pop_placing_agents)
        self.hof_per_generation.append(copy.deepcopy(self.hof.items))
        self.hos_per_generation.append(copy.deepcopy(self.hos.items))

        # --- Elitism: Preserve best chromosomes ---
        elite_agents = tools.selBest(self.pop_placing_agents, self.elite_size)
        elite_chromosomes = [copy.deepcopy(agent.strategy.chromosome) for agent in elite_agents]

        # Get the non-elite agents.
        non_elite_agents = [agent for agent in self.pop_placing_agents if agent not in elite_agents]
        assert len(non_elite_agents) == self.population_size - self.elite_size

        # Apply the selection operator on the non-elite agents.
        # For example, if toolbox.select is defined to perform tournament selection:
        selected_agents = toolbox.select(non_elite_agents, len(non_elite_agents))
        selected_chromosomes = [copy.deepcopy(agent.strategy.chromosome) for agent in selected_agents]

        # Apply Crossover on selected chromosomes (pairwise)
        for i in range(1, len(selected_chromosomes), 2):
            selected_chromosomes[i - 1], selected_chromosomes[i] = toolbox.mate_chromosome(
                selected_chromosomes[i - 1], selected_chromosomes[i]
            )

        # Apply Mutation on the selected chromosomes
        for i in range(len(selected_chromosomes)):
            selected_chromosomes[i] = toolbox.mutate_chromosome(selected_chromosomes[i])[0]

        # Combine the updated non-elite chromosomes with the elite chromosomes.
        new_chromosomes = selected_chromosomes + elite_chromosomes

        # Update the stored chromosome list.
        self.population_chromosomes = new_chromosomes

        # Finally, update each agent’s chromosome using the new chromosome list.
        for agent, new_chrom in zip(self.pop_placing_agents, self.population_chromosomes):
            agent.strategy.chromosome = new_chrom
            # Reinitialize or rebuild any internal state that depends on the chromosome.
            agent.new_placements()

    # ------------------- Custom Operators -------------------
    def custom_crossover_placement(self, parent1, parent2):
        offspring1 = []
        offspring2 = []
        board1 = [[0] * self.board_size for _ in range(self.board_size)]
        board2 = [[0] * self.board_size for _ in range(self.board_size)]
        num_ships = len(self.ship_sizes)

        for i in range(num_ships):
            size = self.ship_sizes[i]
            # Offspring 1: Use parent's gene in fixed order.
            for gene in [parent1[i], parent2[i]]:
                if is_gene_valid(
                        board1, gene[0], gene[1], gene[2], self.board_size, size
                ):
                    gene1 = gene
                    break
            else:
                gene1 = random_valid_gene(board1, self.board_size, size)
            offspring1.append(gene1)
            mark_board(board1, gene1, size)

            # Offspring 2: Try parent's gene in reversed order.
            for gene in [parent2[i], parent1[i]]:
                if is_gene_valid(
                        board2, gene[0], gene[1], gene[2], self.board_size, size
                ):
                    gene2 = gene
                    break
            else:
                gene2 = random_valid_gene(board2, self.board_size, size)
            offspring2.append(gene2)
            mark_board(board2, gene2, size)

        return offspring1, offspring2

    def custom_mutation_placement(self, individual, indpb):
        """
        Locally mutate an individual's chromosome by making small changes.
        Each gene is first checked for validity relative to the genes already placed;
        if invalid, it is repaired. Otherwise, with probability indpb, the gene is mutated.
        Returns a tuple containing the repaired chromosome.
        """
        repaired = []
        board = [[0] * self.board_size for _ in range(self.board_size)]
        for i, gene in enumerate(individual):
            size = self.ship_sizes[i]
            if not is_gene_valid(
                    board, gene[0], gene[1], gene[2], self.board_size, size
            ):
                new_gene = random_valid_gene(board, self.board_size, size)
            elif random.random() < indpb:
                new_gene = local_mutation_gene(gene, board, self.board_size, size)
                if not is_gene_valid(
                        board, new_gene[0], new_gene[1], new_gene[2], self.board_size, size
                ):
                    new_gene = random_valid_gene(board, self.board_size, size)
            else:
                new_gene = gene

            repaired.append(new_gene)
            mark_board(board, new_gene, size)

        return (repaired,)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # === Static Parameters (Adjustable) ===
    BOARD_SIZE = 10
    SHIP_SIZES = [5, 4, 3, 2, 2]
    POPULATION_SIZE = 50
    ELITE_SIZE = 1
    NUM_GENERATIONS = 50
    TOURNAMENT_SIZE = 3
    MUTPB = 0.2
    # =======================================

    game_manager = GameManager(size=BOARD_SIZE)

    environment = PlacementGeneticAlgorithm(
        game_manager=game_manager,
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        elite_size=ELITE_SIZE,
        MUTPB=MUTPB,
        TOURNAMENT_SIZE=TOURNAMENT_SIZE,
    )
    environment.initialize_placing_population(POPULATION_SIZE)

    search_agents = []
    for i in range(10):
        search_agent = SearchAgent(
            board_size=BOARD_SIZE, strategy="hunt_down", name=f"hunt_down_{i}"
        )
        # Create a new network instance for each model
        # net = ANET(board_size=BOARD_SIZE,
        #           activation="relu",
        #           device="cpu")

        # search_agent = SearchAgent(
        #    board_size=BOARD_SIZE,
        #    strategy="nn_search",
        #    net=net,
        #    optimizer="adam",
        #    name=f"nn_{i}",
        #    lr=0.001,
        # )
        search_agents.append(search_agent)

    # Outer loop for generations of placing agents and search agents
    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generation {gen + 1}/{NUM_GENERATIONS} ===")

        environment.trigger_evaluate_population(search_agents)

        # Run evolution with metrics tracking
        environment.evolve()

