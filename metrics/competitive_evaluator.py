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
        num_agents = int(self.default_num_placement)
        for _ in range(num_agents):
            placement_agents.append(
                PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="random")
            )
            placement_agents.append(
                PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="uniform_spread")
            )
            #placement_agents.append(
            #    PlacementAgent(board_size=self.board_size, ship_sizes=self.ship_sizes, strategy="chromosome", chromosome=[(0, 0, 0), (1, 1, 0), (2, 2, 0)])
            #)
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

        overall_raw = []
        if is_mapping:
            for key, (genome, sa) in search_agents.items():
                data = sums[sa]
                n    = len(placement_agents)

                # 1) collect raw‐moves‐based metric for plotting
                avg_raw = data['raw'] / n
                overall_raw.append(avg_raw)

                # 2) compute weighted composite fitness
                genome.fitness = self.compute_fitness(
                    avg_raw        = avg_raw,
                    avg_accuracy   = data['accuracy'] / n,
                    avg_efficiency = data['sink_eff'] / n,
                    avg_start_ent  = data['start_ent'] / n,
                    avg_end_ent    = data['end_ent'] / n,
                    board_size     = self.board_size,
                    ship_sizes     = self.ship_sizes,
                    w_moves   = 0.5,
                    w_acc     = 0.0,
                    w_eff     = 0.5,
                    w_entropy = 0.0,
                )

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

    def compute_fitness(self, avg_raw,
                        avg_accuracy,
                        avg_efficiency,
                        avg_start_ent,
                        avg_end_ent,
                        board_size,
                        ship_sizes,
                        w_moves,
                        w_acc,
                        w_eff,
                        w_entropy):
        """
        A simplified fitness = weighted sum of:
          - normalized moves  (fewer is better)
          - accuracy          (higher is better)
          - efficiency        (optional)
          - entropy behavior: high start, low end
        """

        # 1) Normalize moves so 0→worst, 1→best
        board_cells = board_size ** 2
        max_raw = board_cells - sum(ship_sizes)
        moves_score = avg_raw / max_raw if max_raw > 0 else 0.0

        # 2) Accuracy is already in [0,1]
        acc_score = avg_accuracy

        # 3) Efficiency inverted (if you want it)
        eff_score = self.sink_eff_score(avg_efficiency, board_size, ship_sizes)

        # 4) Entropy score: high start * low end
        ent_score = avg_start_ent * (1.0 - avg_end_ent)

        # 5) Weighted sum, normalized back into [0,1]
        total_w = w_moves + w_acc + w_eff + w_entropy
        if total_w <= 0:
            return 0.0

        raw = (w_moves * moves_score +
               w_acc * acc_score +
               w_eff * eff_score +
               w_entropy * ent_score)

        return raw / total_w

    def sink_eff_score(self, avg_efficiency, board_size, ship_sizes):
        """
        avg_efficiency = average # moves between first hit and sink across ships
        board_size     = side length of the square board
        ship_sizes     = list of ship-lengths (e.g. [3,3,2])
        """
        # Total cells
        N = board_size**2

        # 1) Best‐case: once you hit a ship of length L, you need exactly (L-1) further hits.
        #    So best_per_ship = L_i - 1.
        #    Best overall avg = mean(L_i - 1).
        best = sum(L - 1 for L in ship_sizes) / len(ship_sizes)

        # 2) Random‐baseline: if you’re just guessing uniformly among N cells,
        #    the expected number of draws until you’ve hit all the remaining (L_i - 1)
        #    cells is roughly N/2 (the mean of a uniform [1…N] draw).
        #    We’ll use N/2 as our “random” benchmark.
        random_baseline = N / 2.0

        # 3) Linearly map [best → random_baseline] to [1.0 → 0.0]
        score = (random_baseline - avg_efficiency) / (random_baseline - best)

        # 4) Clamp to [0,1]
        sink_eff = max(0.0, min(1.0, score))
        return sink_eff

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
                           hos_per_generation):

        """
        Single figure showing up to `max_each` HOF boards (top) and HOS boards (bottom).
        Labels show only generation (e.g., "Gen 3").
        """
        max_each = 10  # Max number of boards to show per row

        def flatten_and_label(hall_list):
            boards, labels = [], []
            for g, gen_hall in enumerate(hall_list, start=1):
                for chrom in gen_hall:
                    boards.append(self.create_board_from_chromosome(chrom))
                    labels.append(f"Gen {g}")
            return boards[:max_each], labels[:max_each]

        hof_boards, hof_labels = flatten_and_label(hof_per_generation)
        hos_boards, hos_labels = flatten_and_label(hos_per_generation)

        total = len(hof_boards) + len(hos_boards)
        cols = min(max_each, total)
        rows = 2  # One row for HOF, one for HOS

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4), squeeze=False)

        # Plot HOF on top row
        for i, (bd, lbl) in enumerate(zip(hof_boards, hof_labels)):
            ax = axes[0][i]
            ax.imshow(bd, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(lbl, fontsize=8)
            ax.axis("off")

        # Plot HOS on bottom row
        for i, (bd, lbl) in enumerate(zip(hos_boards, hos_labels)):
            ax = axes[1][i]
            ax.imshow(bd, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(lbl, fontsize=8)
            ax.axis("off")

        # Hide unused subplots
        for r in range(rows):
            for c in range(len(hof_boards if r == 0 else hos_boards), cols):
                axes[r][c].axis("off")

        fig.suptitle("HOF (Top) vs HOS (Bottom)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
