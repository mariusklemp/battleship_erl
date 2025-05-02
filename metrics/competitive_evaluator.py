# metrics/competitive_evaluator.py
import math

import numpy as np
import matplotlib.pyplot as plt
from game_logic.placement_agent import PlacementAgent


class CompetitiveEvaluator:
    def __init__(self, game_manager, board_size, ship_sizes, run_ga, default_num_placement=5):
        """
        Initialize the competitive evaluator.

        Parameters:
          game_manager         : Instance of GameManager used to run simulations.
          board_size           : Size of the Battleship board.
          ship_sizes           : List of ship sizes.
          default_num_placement: Number of default placement agents to create if none are provided.
          run_ga               : Boolean flag indicating if GA is used (to update placement agents' fitness).
        """
        self.game_manager = game_manager
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.default_num_placement = default_num_placement
        self.run_ga = run_ga

        # Histories for plotting competitive evaluation over generations.
        self.placement_eval_history = []
        self.search_eval_history = []

    def init_placement_agents(self):
        placement_agents = []
        for _ in range(self.default_num_placement):
            placement_agents.append(
                PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="random")
            )
            placement_agents.append(
                PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="uniform_spread")
            )
        return placement_agents

    def evaluate(self, search_agents, placement_agents=None):
        """
        Evaluate every pairing between placement agents and search agents.
        Supports both NEAT-style mappings (dict) and plain lists of search agents.

        Parameters:
          search_agents    : Either a dict {key: (genome, agent)} or a list of search agent instances.
          placement_agents : A list of placement agent instances. If None, defaults are created.
        """
        print("Evaluating agents...")
        # Create default placement agents if not provided
        if placement_agents is None:
            placement_agents = self.init_placement_agents()

        # Check search agent input format
        is_mapping = isinstance(search_agents, dict)
        if is_mapping:
            agent_list = [agent for _, (_, agent) in search_agents.items()]
        else:
            agent_list = search_agents

        # Init fitness containers
        placing_fitness = {agent: 0.0 for agent in placement_agents}
        search_fitness = {agent: 0.0 for agent in agent_list}

        num_placement_agents = len(placement_agents)
        num_search_agents = len(agent_list)

        # Evaluate all pairings
        for placement_agent in placement_agents:
            for search_agent in agent_list:
                move_count = self.game_manager.simulate_game(placement_agent, search_agent)

                placing_fitness[placement_agent] += move_count
                search_fitness[search_agent] += (self.board_size ** 2 - move_count)

        # Assign placement agent fitness
        for placement_agent in placement_agents:
            avg_fitness = placing_fitness[placement_agent] / num_search_agents
            if self.run_ga:
                placement_agent.fitness.values = (avg_fitness,)

        # Assign search agent fitness if using NEAT (genomes available)
        if is_mapping:
            for key, (genome, search_agent) in search_agents.items():
                avg_val = np.mean(search_agent.strategy.avg_validation_history)
                avg_acc = np.mean(search_agent.strategy.val_top1_accuracy_history)
                avg_fitness = search_fitness[search_agent] / num_placement_agents
                genome.fitness = avg_fitness

        # FOR PLOTTING: compute correct overall averages from fitness values
        overall_avg_placing = np.mean([
            agent.fitness.values[0] if self.run_ga else placing_fitness[agent] / num_search_agents
            for agent in placement_agents
        ])

        if is_mapping:
            overall_avg_search = np.mean([
                genome.fitness for genome, _ in search_agents.values()
            ])
        else:
            overall_avg_search = np.mean([
                search_fitness[agent] / num_placement_agents for agent in agent_list
            ])

        # Record fitness history
        self.placement_eval_history.append(overall_avg_placing)
        self.search_eval_history.append(overall_avg_search)

        print(f"Placement pop Fitness: {overall_avg_placing:.2f}, Search pop Fitness: {overall_avg_search:.2f}")

        return search_agents, placement_agents

    def create_board_from_chromosome(self, chromosome):
        """
            Create a board (numpy array) from a chromosome.
            Cells covered by a ship are marked as 1.
        """
        board = [[0] * self.board_size for _ in range(self.board_size)]
        for gene, size in zip(chromosome, self.ship_sizes):
            self.mark_board(board, gene, size)
        return np.array(board)

    def mark_board(self, board, gene, size):
        col, row, direction = gene
        if direction == 0:  # horizontal
            for j in range(size):
                board[row][col + j] = 1
        else:  # vertical
            for j in range(size):
                board[row + j][col] = 1

    def plot(self):
        """Plot the competitive evaluation metrics and optionally Hall of Fame and Hall of Shame boards."""
        gens = np.arange(len(self.placement_eval_history))

        # --- Combined Plot: Dual Y-Axis Competitive Fitness ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left axis: Placement
        ax1.plot(gens, self.placement_eval_history, 'o-', color='tab:blue', label='Placement Fitness')
        if len(self.placement_eval_history) > 1:
            p_coef = np.polyfit(gens, self.placement_eval_history, 1)
            p_line = np.poly1d(p_coef)(gens)
            ax1.plot(gens, p_line, '--', color='tab:blue', alpha=0.7, label='Placement Trend')
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Placement Fitness", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Right axis: Search
        ax2 = ax1.twinx()
        ax2.plot(gens, self.search_eval_history, 'o-', color='tab:green', label='Search Fitness')
        if len(self.search_eval_history) > 1:
            s_coef = np.polyfit(gens, self.search_eval_history, 1)
            s_line = np.poly1d(s_coef)(gens)
            ax2.plot(gens, s_line, '--', color='tab:green', alpha=0.7, label='Search Trend')
        ax2.set_ylabel("Search Fitness", color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title("Competitive Evaluation Metrics per Generation")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

        # --- Placement Fitness Only ---
        plt.figure(figsize=(8, 5))
        plt.plot(gens, self.placement_eval_history, 'o-', color='tab:blue', label='Placement Fitness')
        if len(self.placement_eval_history) > 1:
            plt.plot(gens, p_line, '--', color='tab:blue', alpha=0.7, label='Placement Trend')
        plt.xlabel("Generation")
        plt.ylabel("Placement Fitness")
        plt.title("Placement Fitness over Generations")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Search Fitness Only ---
        plt.figure(figsize=(8, 5))
        plt.plot(gens, self.search_eval_history, 'o-', color='tab:green', label='Search Fitness')
        if len(self.search_eval_history) > 1:
            plt.plot(gens, s_line, '--', color='tab:green', alpha=0.7, label='Search Trend')
        plt.xlabel("Generation")
        plt.ylabel("Search Fitness")
        plt.title("Search Fitness over Generations")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_hall_frequencies(self,
                              hof_per_generation,
                              hos_per_generation,
                              board_size):
        # initialize accumulators
        hof_acc = np.zeros((board_size, board_size), dtype=float)
        hos_acc = np.zeros_like(hof_acc)
        gens = len(hof_per_generation)
        hof_size = len(hof_per_generation[0])
        hos_size = len(hos_per_generation[0])

        # sum up all HOF boards
        for gen_hof in hof_per_generation:
            for chrom in gen_hof:
                hof_acc += self.create_board_from_chromosome(chrom)

        # sum up all HOS boards
        for gen_hos in hos_per_generation:
            for chrom in gen_hos:
                hos_acc += self.create_board_from_chromosome(chrom)

        # normalize to [0,1]
        hof_freq = hof_acc / (gens * hof_size)
        hos_freq = hos_acc / (gens * hos_size)

        # plot them side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(hof_freq, vmin=0, vmax=1, cmap='viridis')
        ax1.set_title("HOF Occupancy Frequency")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(hos_freq, vmin=0, vmax=1, cmap='viridis')
        ax2.set_title("HOS Occupancy Frequency")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046)

        plt.suptitle("Across All Generations")
        plt.tight_layout()
        plt.show()

    def plot_halls_wrapped(self,
                           hof_per_generation,
                           hos_per_generation,
                           board_size,
                           max_per_row=10,
                           max_total=20):
        """
        Two figures with wrapping and a hard cap on total boards shown:
          - All HOF boards, up to `max_per_row` per row, with at most `max_total` boards total.
          - All HOS boards likewise.
        """

        def flatten_and_label(hall_list, hall_name):
            boards, labels = [], []
            for g, gen_hall in enumerate(hall_list, start=1):
                for i, chrom in enumerate(gen_hall, start=1):
                    boards.append(self.create_board_from_chromosome(chrom))
                    labels.append(f"G{g}-{hall_name}{i}")
            return boards, labels

        def plot_wrapped(boards, labels, title):
            # --- subsample if over max_total ---
            n = len(boards)
            if n > max_total:
                step = math.ceil(n / max_total)
                boards = boards[::step]
                labels = labels[::step]
                n = len(boards)

            if n == 0:
                print(f"No boards to show for {title}")
                return

            rows = math.ceil(n / max_per_row)
            cols = max_per_row
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(2 * cols, 2 * rows),
                                     squeeze=False)

            for idx, (bd, lbl) in enumerate(zip(boards, labels)):
                r, c = divmod(idx, max_per_row)
                ax = axes[r][c]
                ax.imshow(bd, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(lbl, fontsize=8)
                ax.axis("off")

            # hide any leftover subplots
            for idx in range(n, rows * cols):
                r, c = divmod(idx, max_per_row)
                axes[r][c].axis("off")

            plt.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

        # Hall of Fame
        hof_boards, hof_labels = flatten_and_label(hof_per_generation, "HOF#")
        plot_wrapped(hof_boards, hof_labels, "All Generations: Hall of Fame")

        # Hall of Shame
        hos_boards, hos_labels = flatten_and_label(hos_per_generation, "HOS#")
        plot_wrapped(hos_boards, hos_labels, "All Generations: Hall of Shame")
