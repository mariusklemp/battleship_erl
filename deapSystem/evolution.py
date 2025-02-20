import random
from functools import partial
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import base, creator, tools
import matplotlib.pyplot as plt
import numpy as np

import visualize

from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent

from .helpers import is_gene_valid, mark_board, random_valid_gene, local_mutation_gene

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
    def __init__(
        self,
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
        self.hof = tools.HallOfFame(maxsize=5)
        self.hos = HallOfShame(maxsize=5)
        self.avg_fitness_over_gens = []
        self.diversity_over_gens = []
        self.sparsity_over_gens = []
        self.percent_vertical_over_gens = []  # Only store vertical percentage.

        # Set up game environment.
        self.game_manager = GameManager(size=self.board_size)

        # Initialize population.
        self.pop_placing_agents = self.initialize_placing_population(
            self.population_size
        )

        # Register DEAP operators with our custom methods.
        toolbox.register("mate_chromosome", self.custom_crossover_placement)
        toolbox.register(
            "mutate_chromosome",
            partial(self.custom_mutation_placement, indpb=MUTPB),
        )
        toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # ------------------- Helper to Create Board from Chromosome -------------------
    def create_board_from_chromosome(self, chromosome):
        """
        Create a board (numpy array) from a chromosome.
        Cells covered by a ship are marked as 1.
        """
        board = [[0] * self.board_size for _ in range(self.board_size)]
        for gene, size in zip(chromosome, self.ship_sizes):
            mark_board(board, gene, size)
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

    # ------------------- Helper: Compute Individual Sparsity -------------------
    def compute_individual_sparsity(self, chromosome):
        """
        Compute sparsity for an individual as the average pairwise Manhattan distance
        between the centers of each ship (ignoring orientation).
        """
        positions = [(gene[0], gene[1]) for gene in chromosome]
        if len(positions) < 2:
            return 0
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = abs(positions[i][0] - positions[j][0]) + abs(
                    positions[i][1] - positions[j][1]
                )
                dists.append(d)
        return np.mean(dists)

    # ------------------- Helper: Compute Orientation Percentages -------------------
    def compute_population_orientation(self, population):
        """
        Compute overall vertical/horizontal percentages in the population.
        Assumes gene[2]==1 indicates vertical and 0 indicates horizontal.
        """
        total_ships = 0
        vertical_count = 0
        for agent in population:
            for gene in agent.strategy.chromosome:
                total_ships += 1
                if gene[2] == 1:
                    vertical_count += 1
        if total_ships == 0:
            return 0, 0
        percent_vertical = vertical_count / total_ships * 100
        percent_horizontal = 100 - percent_vertical
        return percent_vertical, percent_horizontal

    # ------------------- Shared Sampling: Opponent Pool -------------------
    def get_opponent_pool(self, n):
        """
        Build a diverse set of opponents.
        """
        opponents = []
        # Hunt_down strategy opponent.
        for i in range(n):
            search_agent_hunt_down = SearchAgent(
                board_size=self.board_size,
                strategy="hunt_down",
                name=f"hunt_down_{i}",  # Add a unique name for each hunt_down agent
            )
            opponents.append(search_agent_hunt_down)

        return opponents

    # ------------------- Game Simulation and Evaluation -------------------
    def simulate_game(self, game_manager, placing_agent, search_agent):
        """Simulate a Battleship game and return the move count, hits, and misses."""
        current_state = game_manager.initial_state(placing=placing_agent)
        # Count total hits and misses at the end instead of tracking during the game
        while not game_manager.is_terminal(current_state):
            move = search_agent.strategy.find_move(current_state)
            current_state = game_manager.next_state(current_state, move)

        # Count hits and misses from the final board state
        # board[1] contains hits (1s)
        # board[2] contains misses (1s)
        hits = sum(square == 1 for square in current_state.board[1])
        misses = sum(square == 1 for square in current_state.board[2])

        return current_state.move_count, hits, misses

    def evaluate_population(
        self, population_placing, game_manager, search_agents, search_metrics=None
    ):
        """
        Evaluate each placing agent by simulating games against a pool of opponents
        and assigning the average performance as fitness.
        """
        opponents = search_agents
        for placing_agent in population_placing:
            fitness_scores = []
            for opp in opponents:
                moves, hits, misses = self.simulate_game(
                    game_manager, placing_agent, opp
                )
                fitness_scores.append(moves)
                # Record search agent metrics if tracker is provided
                if search_metrics is not None:
                    search_metrics.record_game(opp, moves, hits, misses)
            placing_agent.fitness.values = (sum(fitness_scores) / len(fitness_scores),)

    def initialize_placing_population(self, n):
        """Initialize the placing agent population."""
        population = []
        for _ in range(n):
            placing_agent = creator.IndividualPlacementAgent(
                self.board_size, self.ship_sizes, strategy="chromosome"
            )
            population.append(placing_agent)
        return population

    # ------------------- Additional Helper for Diversity -------------------
    def compute_average_pairwise_distance(self, population):
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                d = self.chromosome_distance(
                    population[i].strategy.chromosome, population[j].strategy.chromosome
                )
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
    def plot_metrics(self):
        generations = np.arange(len(self.avg_moves_over_gens))

        # Figure 1: Average Fitness with Regression Line.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            generations, self.avg_moves_over_gens, marker="o", label="Average Fitness"
        )
        if len(self.avg_moves_over_gens) > 1:
            coeffs = np.polyfit(generations, self.avg_moves_over_gens, 1)
            poly_eqn = np.poly1d(coeffs)
            ax.plot(generations, poly_eqn(generations), "r--", label="Regression Line")
        ax.set_title("Average Fitness per Generation")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Move Count")
        ax.legend()
        plt.show()

        # Figure 2: Population Diversity Over Generations.
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        gens_div = np.arange(len(self.diversity_over_gens))
        ax2.plot(gens_div, self.diversity_over_gens, marker="o")
        ax2.set_title("Population Diversity Over Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Average Pairwise Distance")
        plt.show()

        # Figure 3: Board Occupancy Heatmap.
        if len(self.boards_over_generations) > 20:
            step = len(self.boards_over_generations) // 20
            boards_to_plot = self.boards_over_generations[::step]
            gen_labels = np.arange(0, len(self.boards_over_generations), step)
        else:
            boards_to_plot = self.boards_over_generations
            gen_labels = np.arange(len(self.boards_over_generations))
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
            axs[j].axis("off")
        fig3.suptitle("Ship Placement Frequency per Generation")
        plt.tight_layout()
        plt.show()

        # Figure 4: Centroid Evolution of Ship Placements.
        centroids = []
        for board in self.boards_over_generations:
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
        ax4.plot(
            centroids[:, 1], centroids[:, 0], marker="o", linestyle="-", color="magenta"
        )
        ax4.set_title("Centroid Evolution of Ship Placements")
        ax4.set_xlabel("Column")
        ax4.set_ylabel("Row")
        for i, (row, col) in enumerate(centroids):
            ax4.text(col, row, str(i), color="black", ha="left", va="bottom")
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
                axs6[i].set_title(
                    f"Worst {i}\nFitness = {worst_ind.fitness.values[0]:.2f}"
                )
                axs6[i].set_xticks(range(self.board_size))
                axs6[i].set_yticks(range(self.board_size))
            fig6.suptitle("Worst Boards (Hall of Shame)")
            plt.show()

        # Figure 7: Sparsity Over Generations.
        if len(self.sparsity_over_gens) > 0:
            fig7, ax7 = plt.subplots(figsize=(6, 5))
            gens_sparse = np.arange(len(self.sparsity_over_gens))
            ax7.plot(gens_sparse, self.sparsity_over_gens, marker="o")
            ax7.set_title("Average Board Sparsity per Generation")
            ax7.set_xlabel("Generation")
            ax7.set_ylabel("Average Intra-individual Ship Distance")
            plt.show()

        # Figure 8: Orientation Percentages Over Generations.
        if len(self.percent_vertical_over_gens) > 0:
            fig8, ax8 = plt.subplots(figsize=(6, 5))
            gens_orient = np.arange(len(self.percent_vertical_over_gens))
            verticals = np.array(self.percent_vertical_over_gens)  # vertical % stored

            # Plot the vertical percentage line.
            ax8.plot(
                gens_orient,
                verticals,
                marker="o",
                color="blue",
                label="Orientation (%)",
            )

            # Draw a horizontal threshold line at 50%.
            ax8.axhline(50, color="red", linestyle="--", label="50% Threshold")

            # Optionally, fill areas with different colors to visually indicate superiority.
            # When vertical % is above 50, vertical placements are superior.
            ax8.fill_between(
                gens_orient,
                verticals,
                50,
                where=(verticals >= 50),
                color="blue",
                alpha=0.2,
                interpolate=True,
                label="Vertical Superior",
            )
            # When vertical % is below 50, horizontal placements are superior.
            ax8.fill_between(
                gens_orient,
                verticals,
                50,
                where=(verticals < 50),
                color="green",
                alpha=0.2,
                interpolate=True,
                label="Horizontal Superior",
            )

            ax8.set_title(
                "Ship Orientation Over Generations\n(Above 50%: Vertical; Below 50%: Horizontal)"
            )
            ax8.set_xlabel("Generation")
            ax8.set_ylabel("Orientation Percentage (%)")
            ax8.set_ylim(0, 100)
            ax8.legend()
            plt.show()

    # ------------------- Evolutionary Process -------------------
    def evolve(self, gen, search_agents, search_metrics=None):
        print(f"\n-------------- Generation {gen} --------------")
        # Evaluate population using shared sampling.
        print("Evaluating population...")
        toolbox.evaluate_population(
            self.pop_placing_agents, self.game_manager, search_agents, search_metrics
        )

        # Record average fitness.
        avg_moves = sum(
            agent.fitness.values[0] for agent in self.pop_placing_agents
        ) / len(self.pop_placing_agents)
        self.avg_moves_over_gens.append(avg_moves)
        print(f"Average moves in Generation {gen}: {avg_moves}")

        # Record board overlay for this generation.
        avg_board = self.record_generation_board(self.pop_placing_agents)
        self.boards_over_generations.append(avg_board)

        # Compute and record population diversity.
        diversity = self.compute_average_pairwise_distance(self.pop_placing_agents)
        self.diversity_over_gens.append(diversity)

        # Compute and record sparsity.
        sparsity_list = [
            self.compute_individual_sparsity(agent.strategy.chromosome)
            for agent in self.pop_placing_agents
        ]
        avg_sparsity = np.mean(sparsity_list)
        self.sparsity_over_gens.append(avg_sparsity)

        # Compute and record orientation percentage (vertical).
        total_ships = 0
        vertical_count = 0
        for agent in self.pop_placing_agents:
            for gene in agent.strategy.chromosome:
                total_ships += 1
                if gene[2] == 1:  # 1 indicates vertical.
                    vertical_count += 1
        percent_vertical = (
            (vertical_count / total_ships * 100) if total_ships > 0 else 0
        )
        self.percent_vertical_over_gens.append(percent_vertical)

        # Update Hall of Fame and Hall of Shame.
        self.hof.update(self.pop_placing_agents)
        self.hos.update(self.pop_placing_agents)

        # --- Elitism: Preserve best individuals ---
        elite = tools.selBest(self.pop_placing_agents, self.elite_size)

        # Selection
        offspring_placing = toolbox.select(
            self.pop_placing_agents, len(self.pop_placing_agents)
        )
        offspring_placing = list(map(toolbox.clone, offspring_placing))

        # Crossover
        print("\n--- Crossover Step ---")
        for i in range(1, len(offspring_placing), 2):
            p1 = offspring_placing[i - 1].strategy.chromosome
            p2 = offspring_placing[i].strategy.chromosome

            (
                offspring_placing[i - 1].strategy.chromosome,
                offspring_placing[i].strategy.chromosome,
            ) = toolbox.mate_chromosome(p1, p2)

        # Mutation
        print("\n--- Mutation Step ---")
        for i in range(len(offspring_placing)):
            original = offspring_placing[i].strategy.chromosome
            mutated = toolbox.mutate_chromosome(original)[0]
            offspring_placing[i].strategy.chromosome = mutated

        # Reinitialize individuals with updated chromosomes.
        offspring_placing = [
            creator.IndividualPlacementAgent(
                self.board_size,
                self.ship_sizes,
                strategy="chromosome",
                chromosome=ind.strategy.chromosome,
            )
            for ind in offspring_placing
        ]

        # --- Merge Elite with Offspring ---
        combined_population = offspring_placing + elite
        self.pop_placing_agents[:] = tools.selBest(
            combined_population, self.population_size
        )

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
    NUM_GENERATIONS = 10
    TOURNAMENT_SIZE = 3
    MUTPB = 0.2
    # =======================================

    environment = Evolution(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        elite_size=ELITE_SIZE,
        MUTPB=MUTPB,
        TOURNAMENT_SIZE=TOURNAMENT_SIZE,
    )

    # Register DEAP operators with our custom methods.
    toolbox.register("mate_chromosome", environment.custom_crossover_placement)
    toolbox.register(
        "mutate_chromosome", partial(environment.custom_mutation_placement, indpb=MUTPB)
    )
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Run evolution.
    environment.evolve()
