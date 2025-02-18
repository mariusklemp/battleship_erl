import random
from functools import partial
from deap import base, creator, tools
import matplotlib.pyplot as plt
import numpy as np

from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent

from helpers import is_gene_valid, mark_board, random_valid_gene, local_mutation_gene

# ---------------------------------------------------------------------
# Define DEAP Fitness and Individual Types
# ---------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPlacementAgent", PlacementAgent, fitness=creator.FitnessMax)

# Global DEAP toolbox
toolbox = base.Toolbox()


# ---------------------------------------------------------------------
# Define a simple HallOfShame class to store worst individuals.
# ---------------------------------------------------------------------
class HallOfShame:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.items = []

    def update(self, population):
        # Select the worst individuals from the population
        self.items = tools.selWorst(population, self.maxsize)


class Evolution:
    def __init__(self, board_size, ship_sizes, population_size, num_generations,
                 mcts_simulations, mcts_exploration, elite_size=1):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.population_size = population_size
        self.num_generations = num_generations
        self.mcts_simulations = mcts_simulations
        self.mcts_exploration = mcts_exploration
        self.elite_size = elite_size
        self.boards_over_generations = []  # average board overlay per generation
        toolbox.register("evaluate_population", self.evaluate_population)
        self.hof = tools.HallOfFame(maxsize=5)
        self.hos = HallOfShame(maxsize=5)  # Hall of Shame for worst individuals

        # New metrics for visualization:
        self.avg_fitness_over_gens = []
        self.diversity_over_gens = []  # average pairwise chromosome distance

    # ------------------- Helper to Create Board from Chromosome -------------------
    def create_board_from_chromosome(self, chromosome):
        """
        Create a board (numpy array) from a chromosome.
        Cells covered by a ship are marked as 1.
        """
        board = [[0] * self.board_size for _ in range(self.board_size)]
        for gene, size in zip(chromosome, self.ship_sizes):
            mark_board(board, gene, self.board_size, size)
        return np.array(board)

    def record_generation_board(self, population):
        """
        For a given population, create an average board overlay.
        Each cell's value is the fraction of individuals that cover that cell.
        """
        board_accum = np.zeros((self.board_size, self.board_size))
        for agent in population:
            board = self.create_board_from_chromosome(agent.strategy.chromosome)
            board_accum += board
        avg_board = board_accum / len(population)
        return avg_board

    # ------------------- Shared Sampling: Opponent Pool -------------------
    def get_opponent_pool(self):
        """
        Build a diverse set of opponents.
        """
        opponents = []
        # Hunt_down strategy opponent.
        search_agent_hunt_down = SearchAgent(board_size=self.board_size, strategy="hunt_down")
        opponents.append(search_agent_hunt_down)
        return opponents

    # ------------------- Game Simulation and Evaluation -------------------
    def simulate_game(self, game_manager, placing_agent, search_agent):
        """Simulate a Battleship game and return the move count."""
        current_state = game_manager.initial_state(placing=placing_agent)
        while not game_manager.is_terminal(current_state):
            move = search_agent.strategy.find_move(current_state)
            current_state = game_manager.next_state(current_state, move)
        return current_state.move_count

    def evaluate_population(self, population_placing, game_manager):
        """
        Evaluate each placing agent by simulating games against a pool of opponents
        and assigning the average performance as fitness.
        """
        opponents = self.get_opponent_pool(game_manager)
        for placing_agent in population_placing:
            fitness_scores = []
            for opp in opponents:
                fitness = 0
                for i in range(5):
                    fitness += self.simulate_game(game_manager, placing_agent, opp)
                fitness_scores.append(fitness / 5)
            # Higher move count is better.
            placing_agent.fitness.values = (sum(fitness_scores) / len(fitness_scores),)

    def initialize_placing_population(self, n):
        """Initialize the placing agent population."""
        population = []
        for _ in range(n):
            placing_agent = creator.IndividualPlacementAgent(
                self.board_size, self.ship_sizes, strategy="chromosome")
            population.append(placing_agent)
        return population

    # ------------------- Additional Helper for Diversity -------------------
    def compute_average_pairwise_distance(self, population):
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                d = self.chromosome_distance(population[i].strategy.chromosome,
                                             population[j].strategy.chromosome)
                distances.append(d)
        return np.mean(distances) if distances else 0

    def chromosome_distance(self, chrom1, chrom2):
        """
        Compute a distance between two chromosomes.
        """
        distance = 0
        for gene1, gene2 in zip(chrom1, chrom2):
            distance += abs(gene1[0] - gene2[0]) + abs(gene1[1] - gene2[1])
            if gene1[2] != gene2[2]:
                distance += 1
        return distance

    # ------------------- Plotting Metrics -------------------
    def plot_metrics(self, avg_fitness, board_over_gens):
        generations = np.arange(len(avg_fitness))

        # Figure 1: Average Fitness with Regression Line.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(generations, avg_fitness, marker='o', label='Average Fitness')
        if len(avg_fitness) > 1:
            coeffs = np.polyfit(generations, avg_fitness, 1)
            poly_eqn = np.poly1d(coeffs)
            ax.plot(generations, poly_eqn(generations), 'r--', label='Regression Line')
        ax.set_title("Average Fitness per Generation")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Move Count")
        ax.legend()
        plt.show()

        # Figure 2: Population Diversity Over Generations.
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        gens_div = np.arange(len(self.diversity_over_gens))
        ax2.plot(gens_div, self.diversity_over_gens, marker='o')
        ax2.set_title("Population Diversity Over Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Average Pairwise Distance")
        plt.show()

        # Figure 3: Board Occupancy Heatmap.
        if len(board_over_gens) > 20:
            step = len(board_over_gens) // 20
            boards_to_plot = board_over_gens[::step]
            gen_labels = np.arange(0, len(board_over_gens), step)
        else:
            boards_to_plot = board_over_gens
            gen_labels = np.arange(len(board_over_gens))
        num_boards = len(boards_to_plot)
        cols = 5
        rows = (num_boards // cols) + (num_boards % cols > 0)
        fig3, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axs = axs.flatten()
        for i, board in enumerate(boards_to_plot):
            im = axs[i].imshow(board, cmap="viridis", vmin=0, vmax=1)
            axs[i].set_title(f"Gen {gen_labels[i]}")
            axs[i].set_xticks(range(self.board_size))
            axs[i].set_yticks(range(self.board_size))
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        fig3.suptitle("Ship Placement Frequency per Generation")
        plt.tight_layout()
        plt.show()

        # Figure 4: Centroid Evolution of Ship Placements.
        centroids = []
        for board in board_over_gens:
            total = board.sum()
            if total == 0:
                centroids.append((np.nan, np.nan))
            else:
                rows_idx, cols_idx = np.indices(board.shape)
                centroid_row = (rows_idx * board).sum() / total
                centroid_col = (cols_idx * board).sum() / total
                centroids.append((centroid_row, centroid_col))
        centroids = np.array(centroids)
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.plot(centroids[:, 1], centroids[:, 0], marker='o', linestyle='-', color='magenta')
        ax4.set_title("Centroid Evolution of Ship Placements")
        ax4.set_xlabel("Column")
        ax4.set_ylabel("Row")
        for i, (row, col) in enumerate(centroids):
            ax4.text(col, row, str(i), color='black', ha='left', va='bottom')
        plt.show()

        # Figure 5: Hall of Fame Boards.
        if len(self.hof) > 0:
            num_hof = len(self.hof)
            fig5, axs5 = plt.subplots(1, num_hof, figsize=(5 * num_hof, 5))
            if num_hof == 1:
                axs5 = [axs5]
            for i, hof_ind in enumerate(self.hof):
                board = self.create_board_from_chromosome(hof_ind.strategy.chromosome)
                im = axs5[i].imshow(board, cmap="viridis", vmin=0, vmax=1)
                axs5[i].set_title(f"HOF {i}\nFitness = {hof_ind.fitness.values[0]:.2f}")
                axs5[i].set_xticks(range(self.board_size))
                axs5[i].set_yticks(range(self.board_size))
            fig5.suptitle("Hall of Fame Boards")
            plt.show()

        # Figure 6: Worst Boards (Hall of Shame).
        if len(self.hos.items) > 0:
            num_worst = len(self.hos.items)
            fig6, axs6 = plt.subplots(1, num_worst, figsize=(5 * num_worst, 5))
            if num_worst == 1:
                axs6 = [axs6]
            for i, worst_ind in enumerate(self.hos.items):
                board = self.create_board_from_chromosome(worst_ind.strategy.chromosome)
                im = axs6[i].imshow(board, cmap="viridis", vmin=0, vmax=1)
                axs6[i].set_title(f"Worst {i}\nFitness = {worst_ind.fitness.values[0]:.2f}")
                axs6[i].set_xticks(range(self.board_size))
                axs6[i].set_yticks(range(self.board_size))
            fig6.suptitle("Worst Boards (Hall of Shame)")
            plt.show()

    # ------------------- Evolutionary Process -------------------
    def evolve(self):
        # Set up game environment.
        game_manager = GameManager(size=self.board_size)

        # Initialize population.
        pop_placing_agents = self.initialize_placing_population(self.population_size)
        print(f"Initialized population with {len(pop_placing_agents)} individuals.")
        avg_moves_over_gens = []

        for gen in range(self.num_generations):
            print(f"\n-------------- Generation {gen} --------------")
            for i, agent in enumerate(pop_placing_agents):
                print(f"Agent {i} initial chromosome:")
                agent.show_ships()

            # Evaluate population using shared sampling.
            print("Evaluating population...")
            toolbox.evaluate_population(pop_placing_agents, game_manager)
            for i, agent in enumerate(pop_placing_agents):
                print(f"Agent {i} raw fitness (average moves): {agent.fitness.values}")

            # Record average fitness.
            avg_moves = sum(agent.fitness.values[0] for agent in pop_placing_agents) / len(pop_placing_agents)
            avg_moves_over_gens.append(avg_moves)
            print(f"Average moves in Generation {gen}: {avg_moves}")

            # Record board overlay for this generation.
            avg_board = self.record_generation_board(pop_placing_agents)
            self.boards_over_generations.append(avg_board)

            # Compute and record population diversity.
            diversity = self.compute_average_pairwise_distance(pop_placing_agents)
            self.diversity_over_gens.append(diversity)

            # Update Hall of Fame and Hall of Shame.
            self.hof.update(pop_placing_agents)
            self.hos.update(pop_placing_agents)
            print("Hall of Fame individuals (best so far):")
            for hof_ind in self.hof:
                print(hof_ind.strategy.chromosome, hof_ind.fitness.values)
            print("Hall of Shame individuals (worst so far):")
            for shame_ind in self.hos.items:
                print(shame_ind.strategy.chromosome, shame_ind.fitness.values)

            # --- Elitism: Preserve best individuals ---
            elite = tools.selBest(pop_placing_agents, self.elite_size)
            print("Elite individuals (preserved):")
            for e in elite:
                print(e.strategy.chromosome, e.fitness.values)

            # Selection.
            print("\n--- Selection Step ---")
            offspring_placing = toolbox.select(pop_placing_agents, len(pop_placing_agents))
            print("Selected Offspring (before cloning):", offspring_placing)
            offspring_placing = list(map(toolbox.clone, offspring_placing))
            print("Selected Offspring (after cloning):")
            for i, ind in enumerate(offspring_placing):
                print(f"Offspring {i}: {ind.strategy.chromosome}")

            # Crossover.
            print("\n--- Crossover Step ---")
            for i in range(1, len(offspring_placing), 2):
                p1 = offspring_placing[i - 1].strategy.chromosome
                p2 = offspring_placing[i].strategy.chromosome
                print(f"Before Crossover - Pair {i - 1} & {i}:")
                print(f"  Parent 1: {p1}")
                print(f"  Parent 2: {p2}")
                offspring_placing[i - 1].strategy.chromosome, offspring_placing[i].strategy.chromosome = \
                    toolbox.mate_chromosome(p1, p2)
                print(f"After Crossover - Pair {i - 1} & {i}:")
                print(f"  Offspring 1: {offspring_placing[i - 1].strategy.chromosome}")
                print(f"  Offspring 2: {offspring_placing[i].strategy.chromosome}")

            # Mutation.
            print("\n--- Mutation Step ---")
            for i in range(len(offspring_placing)):
                original = offspring_placing[i].strategy.chromosome
                mutated = toolbox.mutate_chromosome(original)[0]
                offspring_placing[i].strategy.chromosome = mutated
                print(f"Agent {i} before mutation: {original}")
                print(f"Agent {i} after mutation:  {mutated}")

            # Reinitialize individuals with updated chromosomes.
            print("\n--- Reinitialization Step ---")
            offspring_placing = [
                creator.IndividualPlacementAgent(self.board_size, self.ship_sizes, strategy="chromosome",
                                                 chromosome=ind.strategy.chromosome)
                for ind in offspring_placing
            ]
            for i, ind in enumerate(offspring_placing):
                print(f"Reinitialized Agent {i}: {ind.strategy.chromosome}")
                ind.show_ships()

            # --- Merge Elite with Offspring ---
            combined_population = offspring_placing + elite
            pop_placing_agents[:] = tools.selBest(combined_population, self.population_size)
            print("\nNew Population:")
            for i, agent in enumerate(pop_placing_agents):
                print(f"Agent {i}: {agent.strategy.chromosome} Fitness: {agent.fitness.values}")

        # Plot metrics at the end.
        self.plot_metrics(avg_moves_over_gens, self.boards_over_generations)

    # ------------------- Custom Operators -------------------
    def custom_crossover_placement(self, parent1, parent2):
        """
        Perform crossover between two parent chromosomes ensuring valid placements.
        Each gene is a tuple (col, row, direction) for a ship with size from self.ship_sizes.
        Returns two offspring chromosomes.
        """
        offspring1 = []
        offspring2 = []
        board1 = [[0] * self.board_size for _ in range(self.board_size)]
        board2 = [[0] * self.board_size for _ in range(self.board_size)]
        num_ships = len(self.ship_sizes)

        for i in range(num_ships):
            size = self.ship_sizes[i]
            # Offspring 1: Try parent's gene in random order.
            candidates1 = [parent1[i], parent2[i]]
            random.shuffle(candidates1)
            gene1 = None
            for gene in candidates1:
                if is_gene_valid(board1, gene[0], gene[1], gene[2], self.board_size, size):
                    gene1 = gene
                    break
            if gene1 is None:
                gene1 = random_valid_gene(board1, self.board_size, size)
            offspring1.append(gene1)
            mark_board(board1, gene1, self.board_size, size)

            # Offspring 2: Try parent's gene in reverse order.
            candidates2 = [parent2[i], parent1[i]]
            random.shuffle(candidates2)
            gene2 = None
            for gene in candidates2:
                if is_gene_valid(board2, gene[0], gene[1], gene[2], self.board_size, size):
                    gene2 = gene
                    break
            if gene2 is None:
                gene2 = random_valid_gene(board2, self.board_size, size)
            offspring2.append(gene2)
            mark_board(board2, gene2, self.board_size, size)

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
            if not is_gene_valid(board, gene[0], gene[1], gene[2], self.board_size, size):
                print(f"Gene {gene} is invalid; repairing.")
                new_gene = random_valid_gene(board, self.board_size, size)
            elif random.random() < indpb:
                print(f"Mutating gene {gene}.")
                new_gene = local_mutation_gene(gene, board, self.board_size, size)
                if not is_gene_valid(board, new_gene[0], new_gene[1], new_gene[2], self.board_size, size):
                    print(f"Locally mutated gene {new_gene} is invalid; repairing.")
                    new_gene = random_valid_gene(board, self.board_size, size)
            else:
                new_gene = gene

            repaired.append(new_gene)
            mark_board(board, new_gene, self.board_size, size)

        print(f"Repaired (mutated) chromosome: {repaired}")
        return repaired,


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # === Static Parameters (Adjustable) ===
    BOARD_SIZE = 10
    SHIP_SIZES = [5, 4, 3, 2, 2]
    POPULATION_SIZE = 100
    ELITE_SIZE = 2
    NUM_GENERATIONS = 10
    MCTS_SIMULATIONS = 50
    MCTS_EXPLORATION = 1.41
    TOURNAMENT_SIZE = 3
    MUTPB = 0.2
    # =======================================

    environment = Evolution(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        mcts_simulations=MCTS_SIMULATIONS,
        mcts_exploration=MCTS_EXPLORATION,
        elite_size=ELITE_SIZE
    )

    # Register DEAP operators with our custom methods.
    toolbox.register("mate_chromosome", environment.custom_crossover_placement)
    toolbox.register("mutate_chromosome", partial(environment.custom_mutation_placement, indpb=MUTPB))
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Run evolution.
    environment.evolve()
