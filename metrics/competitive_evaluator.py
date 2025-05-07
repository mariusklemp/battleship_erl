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
        if placement_agents is None:
            placement_agents = self.init_placement_agents()

        is_mapping = isinstance(search_agents, dict)
        agent_list = [agent for _, (_, agent) in search_agents.items()] if is_mapping else search_agents

        # accumulate for placement GA
        placing_fitness = {pa: 0.0 for pa in placement_agents}

        # accumulate all six metrics for search agents
        sums = {
            sa: {'raw': 0., 'accuracy': 0., 'sink_eff': 0., 'start_ent': 0., 'end_ent': 0.}
            for sa in agent_list
        }

        # simulate
        for pa in placement_agents:
            for sa in agent_list:
                (moves,
                 accuracy,
                 avg_sink_eff,
                 _avg_moves_btwn,  # unused here
                 start_ent,
                 end_ent) = self.game_manager.simulate_game(pa, sa)

                placing_fitness[pa] += moves
                sums[sa]['raw'] += (self.board_size ** 2 - moves)
                sums[sa]['accuracy'] += accuracy
                sums[sa]['sink_eff'] += avg_sink_eff
                sums[sa]['start_ent'] += start_ent
                sums[sa]['end_ent'] += end_ent

        # assign GA fitness
        for pa in placement_agents:
            avg_moves = placing_fitness[pa] / len(agent_list)
            if self.run_ga:
                pa.fitness.values = (avg_moves,)

        # build composite for NEAT
        overall_raw = []
        if is_mapping:
            for key, (genome, sa) in search_agents.items():
                data = sums[sa]
                n = len(placement_agents)

                # averages
                avg_raw = data['raw'] / n
                avg_acc = data['accuracy'] / n
                avg_eff = data['sink_eff'] / n
                avg_start = data['start_ent'] / n
                avg_end = data['end_ent'] / n

                overall_raw.append(avg_raw)

                # 2) compute theoretical bounds
                board_cells    = float(self.board_size**2)
                sum_ship_cells = float(sum(self.ship_sizes))

                # — 2) “Raw” is already inverted moves:  raw = board_cells – moves
                max_raw   = board_cells - sum_ship_cells       # worst inverted (i.e. best moves)
                moves_score = avg_raw / max_raw                # now in [0,1]

                # sink-eff: best is finishing each ship immediately, worst same as moves
                max_eff   = board_cells
                eff_score = (max_eff - avg_eff) / max_eff      # in [0,1]

                # 4) shape accuracy (power-law here, but you can swap in a logistic)
                acc_score = avg_acc ** 3


                # Gaussian‐shape start around 1.0
                sigma_start = 0.05
                start_score = math.exp(-((avg_start - 1.0)**2) / (2 * sigma_start**2))

                # Gaussian‐shape end around your sweet-spot 0.6
                center_end = 0.6
                sigma_end  = 0.15
                end_score  = math.exp(-((avg_end - center_end)**2) / (2 * sigma_end**2))

                # Both are in (0,1]; if either collapses → near 0
                entropy_score = start_score * end_score


                # 6) Re‐combine with a single entropy weight
                w = {
                    'moves'  : 0.40,
                    'acc'    : 0.15,
                    'eff'    : 0.15,
                    'entropy': 0.30
                }

                composite = (
                        w['moves']   * moves_score +
                        w['acc']     * acc_score +
                        w['eff']     * eff_score +
                        w['entropy'] * entropy_score
                )

                genome.fitness = composite

                print(f"  start_score = {start_score:.2f}, end_score = {end_score:.2f}")
                print(f"  entropy_combined = {entropy_score:.2f}")
                print(f"  composite fitness = {composite:.3f}")

            # keep plotting on raw move-based fitness
            overall_avg_search = float(np.mean(overall_raw))

        else:
            overall_avg_search = float(np.mean([
                sums[sa]['raw'] / len(placement_agents) for sa in agent_list
            ]))

        # record for your existing plots
        overall_avg_place = np.mean([
            pa.fitness.values[0] if self.run_ga
            else placing_fitness[pa] / len(agent_list)
            for pa in placement_agents
        ])
        self.placement_eval_history.append(overall_avg_place)
        self.search_eval_history.append(overall_avg_search)

        print(f"Placement pop Fitness: {overall_avg_place:.2f}, "
              f"Search pop Fitness (raw): {overall_avg_search:.2f}")
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
