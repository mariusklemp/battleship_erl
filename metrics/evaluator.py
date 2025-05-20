import copy
import re
from abc import abstractmethod, ABC

import numpy as np
import matplotlib.pyplot as plt

from game_logic.search_agent import SearchAgent
from game_logic.placement_agent import PlacementAgent


class BaseEvaluator(ABC):
    """Base class with common functionality for evaluating agents."""

    def __init__(self, game_manager, board_size, ship_sizes, num_evaluation_games=10):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.game_manager = game_manager
        self.num_evaluation_games = num_evaluation_games
        self.generations = []
        self.baseline = {}

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_results(self):
        pass

    @abstractmethod
    def simulate_game(self, game_manager, agent1, agent2):
        pass

    @abstractmethod
    def evaluate(self, agents, generation, baseline_name):
        pass

    @abstractmethod
    def plot_metrics(self):
        pass


class SearchEvaluator(BaseEvaluator):
    """
    Class to evaluate the performance of evolved search agents against baseline opponents.
    Tracks performance metrics over generations.
    """

    def __init__(self, game_manager, board_size, ship_sizes, baselines, num_evaluation_games=10):
        super().__init__(game_manager, board_size, ship_sizes, num_evaluation_games)

        # Create baseline opponents
        for strategy in baselines:
            self.baseline[strategy] = PlacementAgent(board_size=board_size, ship_sizes=ship_sizes, strategy=strategy)

        self.result = {}
        self.hit_accuracy = {}
        self.sink_efficiency = {}
        self.moves_between_hits = {}
        self.start_entropy = {}
        self.end_entropy = {}
        # Metrics for tracking performance over generations
        for strategy in baselines:
            self.result[strategy] = {}
            self.hit_accuracy[strategy] = {}
            self.sink_efficiency[strategy] = {}
            self.moves_between_hits[strategy] = {}
            self.start_entropy[strategy] = {}
            self.end_entropy[strategy] = {}

    def get_results(self):
        """
        Return a deep copy of every per‐baseline, per‐generation metric,
        plus the list of generations in order.
        """
        return {
            "result": copy.deepcopy(self.result),
            "hit_accuracy": copy.deepcopy(self.hit_accuracy),
            "sink_efficiency": copy.deepcopy(self.sink_efficiency),
            "moves_between_hits": copy.deepcopy(self.moves_between_hits),
            "start_entropy": copy.deepcopy(self.start_entropy),
            "end_entropy": copy.deepcopy(self.end_entropy),
            "generations": list(self.generations),
        }

    def reset(self):
        self.generations.clear()
        for attr in ("result", "hit_accuracy", "sink_efficiency",
                     "moves_between_hits", "start_entropy", "end_entropy"):
            d = getattr(self, attr)
            for bs in d:
                d[bs].clear()

    def evaluate(self, search_agents, generation, baseline_name):
        """
        Evaluate the average performance of all search agents in the population
        against a baseline placement agent.
        """
        self.generations.append(generation)

        # Metrics across all agents
        all_moves = []
        all_accuracy = []
        all_sink_efficiency = []
        all_moves_between_hits = []
        all_start_entropy = []
        all_end_entropy = []

        # Use a fresh random baseline each time
        baseline_placement_agent = self.baseline[baseline_name]

        for agent in search_agents:
            for _ in range(self.num_evaluation_games):
                baseline_placement_agent.new_placements()

                # Simulate a game with the current agent
                moves, accuracy, sink_efficiency, moves_between_hits, start_entropy, end_entropy = self.simulate_game(
                    self.game_manager, baseline_placement_agent, agent
                )

                all_moves.append(moves)
                all_accuracy.append(accuracy)
                all_sink_efficiency.append(sink_efficiency)
                all_moves_between_hits.append(moves_between_hits)
                all_start_entropy.append(start_entropy)
                all_end_entropy.append(end_entropy)

        # Compute average metrics across all games and all agents
        avg_moves = np.mean(all_moves)
        avg_accuracy = np.mean(all_accuracy)
        avg_sink_efficiency = np.mean(all_sink_efficiency)
        avg_moves_between_hits = np.mean(all_moves_between_hits)
        avg_start_entropy = np.mean(all_start_entropy)
        avg_end_entropy = np.mean(all_end_entropy)

        # Store the results
        self.result[baseline_name][generation] = avg_moves
        self.hit_accuracy[baseline_name][generation] = avg_accuracy
        self.sink_efficiency[baseline_name][generation] = avg_sink_efficiency
        self.moves_between_hits[baseline_name][generation] = avg_moves_between_hits
        self.start_entropy[baseline_name][generation] = avg_start_entropy
        self.end_entropy[baseline_name][generation] = avg_end_entropy

    def evaluate_final_agent(self, search_agent, num_games=100):
        """
        Evaluate a single trained search agent against all baselines and
        generate a radar chart + print stats.
        """
        all_moves = []
        all_accuracy = []
        all_sink_efficiency = []
        all_moves_between_hits = []

        for baseline_name, placement_agent in self.baseline.items():
            for _ in range(num_games):
                print(f"Evaluating against {baseline_name} placement agent", _)
                placement_agent.new_placements()
                if search_agent.name == "mcts":
                    search_agent.strategy.mcts.root_node = None  # Reset between each game
                result = self.simulate_game(self.game_manager, placement_agent, search_agent)
                moves, acc, sink_eff, mbh, start_e, end_e = result

                all_moves.append(moves)
                all_accuracy.append(acc)
                all_sink_efficiency.append(sink_eff)
                all_moves_between_hits.append(mbh)

        # Average metrics across all games
        metrics = {
            "Avg Moves": np.mean(all_moves),
            "Hit Accuracy": np.mean(all_accuracy),
            "Moves Between Hits": np.mean(all_moves_between_hits),
            "Sink Efficiency": np.mean(all_sink_efficiency),
        }
        return metrics

    def simulate_game(self, game_manager, placement_agent, search_agent):
        """Plays one game with the given search agent and returns game metrics."""
        current_state = game_manager.initial_state(placing=placement_agent)

        # Initialize metrics
        total_moves = 0
        hits = 0
        misses = 0
        last_hit_move = None
        moves_between_hits_list = []
        ship_hit_tracking = {}
        sink_moves = []
        entropy_distributions = []

        while not game_manager.is_terminal(current_state):
            result = search_agent.strategy.find_move(current_state, topp=True)

            # If the result is a tuple, assume it includes probabilities
            if isinstance(result, tuple):
                move, probabilities_np = result
            else:
                move = result
                board_size = self.board_size
                probabilities_np = np.ones(board_size ** 2) / (board_size ** 2)  # Uniform fallback

            # Remove illegal moves from the distrubution
            board_flat = np.array(current_state.board[0]).flatten()
            legal_mask = (board_flat == 0)  # True where move is still allowed
            legal_probs = probabilities_np[legal_mask]
            entropy_distributions.append(legal_probs)

            # Check if move will be a hit before applying it
            is_hit = move in current_state.placing.indexes

            # Process the move
            new_state = game_manager.next_state(current_state, move)

            # Update hit/miss count
            if is_hit:
                hits += 1
                if last_hit_move is not None:
                    moves_between = total_moves - last_hit_move
                    moves_between_hits_list.append(moves_between)
                last_hit_move = total_moves

                # Check if this move sunk a ship
                sunk, ship_size, hit_ship = game_manager.check_ship_sunk(move, new_state.board, current_state.placing)
                if hit_ship:
                    ship_id = tuple(sorted(hit_ship))
                    if ship_id not in ship_hit_tracking:
                        ship_hit_tracking[ship_id] = total_moves
                    if sunk:
                        moves_to_sink = total_moves - ship_hit_tracking[ship_id] + 1
                        sink_moves.append(moves_to_sink)
            else:
                misses += 1

            current_state = new_state
            total_moves += 1

        # === Metrics ===
        accuracy = hits / total_moves if total_moves > 0 else 0
        avg_sink_efficiency = sum(sink_moves) / len(sink_moves) if sink_moves else 0
        avg_moves_between_hits = sum(moves_between_hits_list) / len(
            moves_between_hits_list) if moves_between_hits_list else 0

        # === Entropy Calculation ===
        entropies = [self.calculate_entropy(dist) for dist in entropy_distributions]

        if len(entropies) >= 3:
            start_entropy = np.mean(entropies[:3])
            end_entropy = np.mean(entropies[-3:])
        else:
            # Fallback if fewer than 3 moves
            start_entropy = np.mean(entropies)
            end_entropy = np.mean(entropies)

        return total_moves, accuracy, avg_sink_efficiency, avg_moves_between_hits, start_entropy, end_entropy

    def calculate_entropy(self, distribution):
        """
        Calculate the entropy of a probability distribution.
        Higher entropy means more uniform distribution (less concentrated).
        Lower entropy means more concentrated distribution (more certainty).

        :param distribution: A probability distribution (numpy array)
        :return: The entropy value
        """
        # Filter out zero probabilities to avoid log(0)
        probabilities = distribution[distribution > 0]

        if len(probabilities) == 0:
            return 0

        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(distribution))
        if max_entropy == 0:
            return 0
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def plot_metrics(self):
        """Plot metrics as line charts for better readability."""
        grouped_line_metrics = {
            "Distribution Entropy": {
                "metrics": ("Start Entropy", "End Entropy"),
                "colors": ("lightblue", "steelblue"),
            },
        }

        single_line_metrics = {
            "Sink Efficiency": ("sink_efficiency", "salmon"),
            "Moves Between Hits": ("moves_between_hits", "orange"),
        }

        line_metrics = {
            "Move Count": self.result,
            "Hit Accuracy": self.hit_accuracy,
        }

        def average_across_baselines(data_dict):
            combined = {}
            count = {}
            for baseline in data_dict:
                for gen, value in data_dict[baseline].items():
                    combined[gen] = combined.get(gen, 0) + value
                    count[gen] = count.get(gen, 0) + 1
            for gen in combined:
                combined[gen] /= count[gen]
            return combined

        # === Grouped line metrics (start vs end entropy) ===
        for group_title, meta in grouped_line_metrics.items():
            metric1, metric2 = meta["metrics"]
            color1, color2 = meta["colors"]

            data1 = getattr(self, metric1.lower().replace(" ", "_"))
            data2 = getattr(self, metric2.lower().replace(" ", "_"))

            avg_data1 = average_across_baselines(data1)
            avg_data2 = average_across_baselines(data2)

            gens = sorted(set(avg_data1) | set(avg_data2), key=self.gen_key)
            x_labels = [str(g) for g in gens]

            values1 = [avg_data1.get(g, 0) for g in gens]
            values2 = [avg_data2.get(g, 0) for g in gens]

            plt.figure(figsize=(12, 6))
            plt.plot(x_labels, values1, 'o-', label=metric1, color=color1, linewidth=2)
            plt.plot(x_labels, values2, 'o-', label=metric2, color=color2, linewidth=2)

            plt.xticks(rotation=45)
            plt.title(f"Search Agent: {group_title}")
            plt.xlabel("Generation")
            plt.ylabel(group_title)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === Single-line metrics (sink efficiency & moves between hits) ===
        for title, (attr_name, color) in single_line_metrics.items():
            data = getattr(self, attr_name)
            avg_data = average_across_baselines(data)

            gens = sorted(avg_data.keys(), key=self.gen_key)
            x_labels = [str(g) for g in gens]
            values = [avg_data[g] for g in gens]

            plt.figure(figsize=(10, 5))
            plt.plot(x_labels, values, 'o-', color=color, linewidth=2, label=title)

            plt.xticks(rotation=45)
            plt.title(f"Search Agent: {title}")
            plt.xlabel("Generation")
            plt.ylabel(title)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === Line plots for move count and hit accuracy ===
        for metric_name, metric_data in line_metrics.items():
            plt.figure(figsize=(10, 6))
            for baseline, values in metric_data.items():
                gens = sorted(values.keys(), key=self.gen_key)
                x_labels = [str(g) for g in gens]
                y_vals = [values[g] for g in gens]
                plt.plot(x_labels, y_vals, marker="o", label=baseline, linewidth=2)

            plt.xticks(rotation=45)
            plt.title(f"Search Agent: {metric_name}")
            plt.xlabel("Generation")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _collapse_baselines(self, metric_dict):
        """
        metric_dict: { baseline_name: { gen: value, … }, … }
        → returns { gen: [values across baselines], … }
        """
        tmp = {}
        for bs, gen_vals in metric_dict.items():
            for g, v in gen_vals.items():
                tmp.setdefault(g, []).append(v)
        return tmp

    def _collapse_variations(self, list_of_gen2vals):
        """
        list_of_gen2vals: [ {gen: val, …}, … ] for each variation
        → sorted_gens, mean_array, std_array
        """
        all_gens = sorted({g for d in list_of_gen2vals for g in d}, key=self.gen_key)
        A = np.array([[d.get(g, 0.0) for g in all_gens]
                      for d in list_of_gen2vals])
        return all_gens, A.mean(axis=0), A.std(axis=0)

    def aggregate_runs(self, runs):
        agg = {}
        for key in ('start_entropy', 'end_entropy'):
            # step 1: for each run, collapse baselines → gen→mean_across_baselines
            per_run = []
            for r in runs:
                bs_collapsed = self._collapse_baselines(r[key])
                # compute mean across baselines for each gen
                per_run.append({g: np.mean(vs) for g, vs in bs_collapsed.items()})
            # step 2: collapse across variations
            gens, means, stds = self._collapse_variations(per_run)
            agg[key] = {'gens': gens, 'mean': means, 'std': stds}

        # For these preserve per-baseline curves:
        for key in ('sink_efficiency', 'moves_between_hits',
                    'result', 'hit_accuracy'):
            agg[key] = {}
            # runs is list of result-dicts; each r[key] is baseline→{gen→value}
            for bs in runs[0][key].keys():
                # extract for this baseline across all runs
                per_run = [r[key][bs] for r in runs]
                gens, mean, std = self._collapse_variations(per_run)
                agg[key][bs] = {"gens": gens, "mean": mean, "std": std}
        return agg

    def plot_metrics_from_agg(self, stats, experiment: str):
        """
        stats: either
          - a single-agg dict (old behavior), or
          - a dict with exactly one key "solo" (RL), or
          - a dict with keys "solo" and "co_evo" (neat/erl)
        """
        # --- unwrap the RL-only case ---
        if isinstance(stats, dict) and "solo" in stats and "co_evo" not in stats:
            stats = stats["solo"]
            is_nested = False
        elif isinstance(stats, dict) and "solo" in stats and "co_evo" in stats:
            is_nested = True
        else:
            is_nested = False

        # color maps
        base_colors = {
            "random": "#1f77b4",  # blue
            "uniform spread": "#ff7f0e",  # orange
        }
        dark_colors = {
            "random": "#0d3f8b",  # darker blue
            "uniform spread": "#d66000",  # darker orange
        }

        def classify_baseline(name):
            nl = name.lower()
            if "random" in nl:
                return "random"
            if "uniform" in nl or "spread" in nl:
                return "uniform spread"
            return name

        def get_color(baseline, mode):
            key = classify_baseline(baseline)
            if is_nested and mode == "co_evo":
                return dark_colors[key]
            return base_colors[key]

        def get_label(baseline, mode):
            cls = classify_baseline(baseline).title()
            if is_nested:
                if mode == "co_evo":
                    return f"{cls} (Co-evo)"
                else:
                    return cls
            return cls

        # --- 1) Distribution Entropy ---
        fig, ax = plt.subplots(figsize=(12, 6))
        if not is_nested:
            gens = stats['start_entropy']['gens']
            start_m, start_s = stats['start_entropy']['mean'], stats['start_entropy']['std']
            end_m, end_s = stats['end_entropy']['mean'], stats['end_entropy']['std']
            x = np.arange(len(gens))

            ax.plot(
                x, start_m,
                marker='o', linestyle='-', linewidth=3, markersize=6,
                label='Start Entropy',
                color=base_colors["random"]
            )
            ax.fill_between(x, start_m - start_s, start_m + start_s,
                            alpha=0.2, color=base_colors["random"])

            ax.plot(
                x, end_m,
                marker='o', linestyle='-', linewidth=3, markersize=6,
                label='End Entropy',
                color=base_colors["uniform spread"]
            )
            ax.fill_between(x, end_m - end_s, end_m + end_s,
                            alpha=0.2, color=base_colors["uniform spread"])
        else:
            styles = {
                "solo": {"linestyle": "-", "color": base_colors["random"]},
                "co_evo": {"linestyle": "--", "color": dark_colors["random"]},
            }
            # plot Start Entropy
            for mode, style in styles.items():
                s = stats[mode]
                gens = s['start_entropy']['gens']
                x = np.arange(len(gens))
                m0, s0 = s['start_entropy']['mean'], s['start_entropy']['std']

                ax.plot(
                    x, m0,
                    marker='o', linewidth=3, markersize=6,
                    linestyle=style["linestyle"],
                    label=get_label("random", mode),
                    color=style["color"]
                )
                ax.fill_between(x, m0 - s0, m0 + s0,
                                alpha=0.2, color=style["color"])

            # plot End Entropy (uniform spread baselines)
            styles_end = {
                "solo": {"linestyle": "-", "color": base_colors["uniform spread"]},
                "co_evo": {"linestyle": "--", "color": dark_colors["uniform spread"]},
            }
            for mode, style in styles_end.items():
                s = stats[mode]
                gens = s['end_entropy']['gens']
                x = np.arange(len(gens))
                m1, s1 = s['end_entropy']['mean'], s['end_entropy']['std']

                ax.plot(
                    x, m1,
                    marker='o', linewidth=3, markersize=6,
                    linestyle=style["linestyle"],
                    label=get_label("uniform spread", mode),
                    color=style["color"]
                )
                ax.fill_between(x, m1 - s1, m1 + s1,
                                alpha=0.2, color=style["color"])

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gens], rotation=45)
        ax.set_title(f"{experiment} — Distribution Entropy", fontsize=14)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.show()

        # --- 2) Four per-baseline metrics (mean ± std) ---
        metrics = [
            ("Move Count", "result"),
            ("Hit Accuracy", "hit_accuracy"),
            ("Moves Between Hits", "moves_between_hits"),
            ("Sink Efficiency", "sink_efficiency"),
        ]

        for title, key in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))

            if not is_nested:
                for baseline, data in stats[key].items():
                    gens = data['gens']
                    m, s = data['mean'], data['std']
                    x = np.arange(len(gens))
                    col = base_colors[classify_baseline(baseline)]
                    ax.plot(
                        x, m,
                        marker='o', linestyle='-', linewidth=3, markersize=6,
                        label=classify_baseline(baseline).title(),
                        color=col
                    )
                    ax.fill_between(x, m - s, m + s, alpha=0.2, color=col)
            else:
                for mode, ls in [("solo", "-"), ("co_evo", "--")]:
                    sdict = stats[mode]
                    for baseline, data in sdict[key].items():
                        gens = data['gens']
                        m, s = data['mean'], data['std']
                        x = np.arange(len(gens))
                        col = get_color(baseline, mode)
                        ax.plot(
                            x, m,
                            marker='o', linestyle=ls, linewidth=3, markersize=6,
                            label=get_label(baseline, mode),
                            color=col
                        )
                        ax.fill_between(x, m - s, m + s, alpha=0.2, color=col)

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gens], rotation=45)
            ax.set_title(f"{experiment} — {title}", fontsize=16)
            ax.set_xlabel("Generation", fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
            plt.tight_layout()
            plt.show()

    def plot_combined_all(self, all_stats):
        """
        Overlay multiple experiments' aggregated stats:
          - One combined entropy plot (start & end together)
          - Individual plots for other metrics
        """
        # --- 1) Combined Start & End Entropy ---
        fig, ax = plt.subplots(figsize=(12, 6))
        example = next(iter(all_stats.values()))
        gens = example['start_entropy']['gens']
        x = np.arange(len(gens))
        for exp_label, stats in all_stats.items():
            # start
            s_mean, s_std = stats['start_entropy']['mean'], stats['start_entropy']['std']
            ax.plot(x, s_mean, 'o--', linewidth=3, markersize=6,
                    label=f"{exp_label} Start", color=None)
            ax.fill_between(x, s_mean - s_std, s_mean + s_std, alpha=0.2)
            # end
            e_mean, e_std = stats['end_entropy']['mean'], stats['end_entropy']['std']
            ax.plot(x, e_mean, 'o-', linewidth=3, markersize=6,
                    label=f"{exp_label} End", color=None)
            ax.fill_between(x, e_mean - e_std, e_mean + e_std, alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gens], rotation=45)
        ax.set_title("Entropy Across Experiments", fontsize=14)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(title="Experiment", loc='upper right', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.show()

        # --- 2) Other Metrics Combined ---
        metrics = [
            ("Move Count", 'result'),
            ("Hit Accuracy", 'hit_accuracy'),
            ("Moves Between Hits", 'moves_between_hits'),
            ("Sink Efficiency", 'sink_efficiency'),
        ]
        for title, key in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            for exp_label, stats in all_stats.items():
                sk = stats[key]
                if 'gens' in sk:
                    gens, m, s = sk['gens'], sk['mean'], sk['std']
                else:
                    first = next(iter(sk.values()))
                    gens = first['gens']
                    all_means = np.vstack([v['mean'] for v in sk.values()])
                    m = all_means.mean(axis=0)
                    s = all_means.std(axis=0)
                x = np.arange(len(gens))
                ax.plot(x, m, 'o-', linewidth=3, markersize=6, label=exp_label)
                ax.fill_between(x, m - s, m + s, alpha=0.2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gens], rotation=45)
            ax.set_title(f"{title} Across Experiments", fontsize=16)
            ax.set_xlabel("Generation", fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend(title="Experiment", loc='upper right', fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plt.show()

    def gen_key(self, x):
        """
            Extract generation number as int from keys like:
              - 'nn_gen20'
              - 'random'   (no digits → fallback to 0 or just return x if you prefer)
              - 5          (already an int)
            """
        if isinstance(x, str):
            m = re.search(r'(\d+)$', x)
            if m:
                return int(m.group(1))
            else:
                # no trailing digits → put these first or last as you like
                return -1
        return int(x)

    def normalize(self, m):
        # Setup
        best_case = sum(self.ship_sizes)  # Total ship tiles
        worst_case = self.board_size ** 2
        expected_best_moves = best_case * 1.5  # Rough estimate for very good play
        best_hit_accuracy = best_case / expected_best_moves  # e.g., ~0.77
        worst_hit_accuracy = best_case / worst_case  # e.g., ~0.1

        def normalize_hit_accuracy(value):
            # Normalize between worst and realistic best
            norm = (value - worst_hit_accuracy) / (best_hit_accuracy - worst_hit_accuracy)
            return max(0, min(1, norm))

        def normalize_avg_moves(avg_moves):
            norm = (avg_moves - best_case) / (worst_case - best_case)
            return max(0, min(1, norm))

        def normalize_sink_efficiency(value):
            best = sum(self.ship_sizes) / len(self.ship_sizes)
            worst = worst_case * 0.7
            norm = 1 - ((value - best) / (worst - best))
            return max(0, min(1, norm))

        def normalize_moves_between_hits(value):
            best = 1.0
            worst = worst_case / sum(self.ship_sizes)
            norm = (value - best) / (worst - best)
            return max(0, min(1, norm))

        normalized_metrics = {
            "Avg Moves": normalize_avg_moves(m["Avg Moves"]),
            "Hit Accuracy": normalize_hit_accuracy(m["Hit Accuracy"]),
            "Moves Between Hits": normalize_moves_between_hits(m["Moves Between Hits"]),
            "Sink Efficiency": normalize_sink_efficiency(m["Sink Efficiency"]),
        }
        return normalized_metrics

    def plot_final_skill_radar_chart(self, labeled_metrics, title="Skill Comparison"):
        """
        Radar chart for multiple agents: supports mean ± std shading.
        Each entry in `labeled_metrics` should be a tuple:
            (label, metrics_mean: dict, metrics_std: dict)
        Baselines can pass a std‐dict of zeros.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n📊 Radar Chart Metrics:")
        for label, mean_m, std_m in labeled_metrics:
            print(f"\n🔹 {label}:")
            for k in mean_m:
                print(f"    {k:<22} mean={mean_m[k]:.3f} ± std={std_m[k]:.3f}")

        categories = ["Avg Moves", "Hit Accuracy", "Moves Between Hits", "Sink Efficiency"]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Set tick positions but not labels (we'll add them manually)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # clear default labels

        # Define custom label placement
        for i, (angle, label) in enumerate(zip(angles[:-1], categories)):
            if i == 0 or i == 2:  # Right (0°) and left (180°) positions
                rotation = 90
                ha = 'center'
            else:
                rotation = 0
                ha = 'center'

            ax.text(
                angle,
                1.1,  # distance from center
                label,
                fontsize=12,
                rotation=rotation,
                rotation_mode='anchor',
                horizontalalignment=ha,
                verticalalignment='center'
            )

        ax.set_yticklabels([])  # hide radial labels
        ax.grid(alpha=0.3)

        for label, mean_m, std_m in labeled_metrics:
            norm_mean = self.normalize(mean_m)
            values = [norm_mean[c] for c in categories] + [norm_mean[categories[0]]]

            low_vals, high_vals = [], []
            for c in categories:
                low = self.normalize({
                    "Avg Moves": mean_m.get("Avg Moves", 0) - std_m.get("Avg Moves", 0),
                    "Hit Accuracy": mean_m.get("Hit Accuracy", 0) - std_m.get("Hit Accuracy", 0),
                    "Moves Between Hits": mean_m.get("Moves Between Hits", 0) - std_m.get("Moves Between Hits", 0),
                    "Sink Efficiency": mean_m.get("Sink Efficiency", 0) - std_m.get("Sink Efficiency", 0),
                })[c]
                high = self.normalize({
                    "Avg Moves": mean_m.get("Avg Moves", 0) + std_m.get("Avg Moves", 0),
                    "Hit Accuracy": mean_m.get("Hit Accuracy", 0) + std_m.get("Hit Accuracy", 0),
                    "Moves Between Hits": mean_m.get("Moves Between Hits", 0) + std_m.get("Moves Between Hits", 0),
                    "Sink Efficiency": mean_m.get("Sink Efficiency", 0) + std_m.get("Sink Efficiency", 0),
                })[c]
                low_vals.append(low)
                high_vals.append(high)

            low_vals += low_vals[:1]
            high_vals += high_vals[:1]

            ll = label.lower()
            if "initial" in ll:
                ls, lw, alpha_fill = 'solid', 3, 0.1
            elif "final" in ll:
                ls, lw, alpha_fill = 'solid', 3, 0.15
            else:
                ls, lw, alpha_fill = 'dotted', 2, 0.08

            ax.plot(angles, values, linestyle=ls, linewidth=lw,
                    marker='o', markersize=6, label=label)
            ax.fill_between(angles, low_vals, high_vals, alpha=alpha_fill)

        ax.set_title(title, fontsize=16, pad=40)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        plt.tight_layout()
        plt.show()



class PlacementEvaluator(BaseEvaluator):
    """
    Class to evaluate the performance of evolved placement agents against baseline opponents.
    Tracks performance metrics over generations.
    """

    def __init__(self, game_manager, board_size, ship_sizes, baselines, num_evaluation_games=10):
        super().__init__(game_manager, board_size, ship_sizes, num_evaluation_games)

        # Create baseline opponents
        for strategy in baselines:
            self.baseline[strategy] = SearchAgent(board_size=board_size, strategy=strategy)

        # Metrics for tracking performance over generations
        self.move_count = {}
        self.sparsity = {}
        self.hit_to_sunk_ratio = {}

        for strategy in baselines:
            self.move_count[strategy] = {}
            self.sparsity[strategy] = {}
            self.hit_to_sunk_ratio[strategy] = {}

        self.sparsity = {}
        self.diversity = {}
        self.orientation = {}
        self.boards_over_generations = {}

    def get_results(self):
        """Return a deep copy of all placement metrics plus generation list."""
        return {
            "move_count": copy.deepcopy(self.move_count),
            "hit_to_sunk_ratio": copy.deepcopy(self.hit_to_sunk_ratio),
            "sparsity": copy.deepcopy(self.sparsity),
            "diversity": copy.deepcopy(self.diversity),
            "orientation": copy.deepcopy(self.orientation),
            "boards_over_generations": copy.deepcopy(self.boards_over_generations),
            "generations": list(self.generations),
        }

    def reset(self):
        """Clear all stored metrics and generation history."""
        self.generations.clear()
        for d in (self.move_count, self.hit_to_sunk_ratio):
            for bs in d:
                d[bs].clear()
        self.sparsity.clear()
        self.diversity.clear()
        self.orientation.clear()
        self.boards_over_generations.clear()

    def evaluate(self, placement_agents, generation, baseline_name):
        """
            Evaluate placement agents against baseline search agents.
            Runs multiple games and averages the results for more reliable metrics.
            """
        self.generations.append(generation)

        # Evaluate against baseline search multiple times
        all_moves = []
        all_hit_to_sunk_ratio = []

        baseline_search_agent = self.baseline[baseline_name]
        for agent in placement_agents:
            for _ in range(self.num_evaluation_games):
                moves, hit_to_sunk_ratio = self.simulate_game(
                    self.game_manager,
                    agent,
                    baseline_search_agent,
                )
                all_moves.append(moves)
                all_hit_to_sunk_ratio.append(hit_to_sunk_ratio)

        # Calculate averages for random search
        avg_moves = np.mean(all_moves)
        avg_hit_ratio = np.mean(all_hit_to_sunk_ratio)

        # Store the average performance
        self.move_count[baseline_name][generation] = avg_moves
        self.hit_to_sunk_ratio[baseline_name][generation] = avg_hit_ratio

        # --- Compute structure-based metrics  ---
        avg_board = self.record_generation_board(placement_agents)
        self.boards_over_generations[generation] = avg_board

        diversity = self.compute_average_pairwise_distance(placement_agents)
        self.diversity[generation] = diversity

        sparsities = [self.compute_individual_sparsity(agent.strategy.chromosome) for agent in placement_agents]
        avg_sparsity = np.mean(sparsities)
        self.sparsity[generation] = avg_sparsity

        # Compute and record orientation percentage (vertical).
        total_ships = 0
        vertical_count = 0
        for agent in placement_agents:
            for gene in agent.strategy.chromosome:
                total_ships += 1
                if gene[2] == 1:  # 1 indicates vertical.
                    vertical_count += 1
        percent_vertical = (
            (vertical_count / total_ships * 100) if total_ships > 0 else 0
        )
        self.orientation[generation] = percent_vertical

    def simulate_game(self, game_manager, placing_agent, search_agent):
        """
        Simulate a Battleship game and return:
          - total_moves: total number of moves taken in the game,
          - avg_moves_per_ship: average number of moves from the first hit on a ship until it is sunk,
          - sparsity: a measure of how spread out the ships are.
        """
        # Initialize game state using the placement agent's board
        current_state = game_manager.initial_state(placing=placing_agent)
        total_moves = 0

        # Dictionaries to record when each ship is first hit and when it gets sunk
        ship_first_hit_move = {}  # key: sorted tuple of ship indexes, value: move count of first hit
        per_ship_moves = []  # list of (ship_key, moves_from_first_hit_to_sunk)

        while not game_manager.is_terminal(current_state):
            total_moves += 1
            # Get move from the search agent strategy (handle tuple vs. single value)
            result = search_agent.strategy.find_move(current_state)
            move, distribution = (result, None) if not isinstance(result, tuple) else result

            # Execute the move: this updates the board and the remaining ships.
            current_state = game_manager.next_state(current_state, move)

            # Check if the move was a hit.
            if move in placing_agent.indexes:
                # Identify the ship that was hit. We iterate over the ships list.
                for ship in placing_agent.ships:
                    if move in ship.indexes:
                        ship_key = tuple(sorted(ship.indexes))
                        # Record the first hit for the ship if not already recorded.
                        if ship_key not in ship_first_hit_move:
                            ship_first_hit_move[ship_key] = total_moves

                        # Check if this hit sinks the ship.
                        sunk, ship_size, hit_ship = game_manager.check_ship_sunk(move, current_state.board,
                                                                                 placing_agent)
                        if sunk:
                            moves_for_ship = total_moves - ship_first_hit_move[ship_key] + 1
                            if ship_key not in [k for (k, _) in per_ship_moves]:
                                per_ship_moves.append((ship_key, moves_for_ship))

        # Compute average moves per sunk ship
        if per_ship_moves:
            avg_moves_per_ship = sum(moves for (_, moves) in per_ship_moves) / len(per_ship_moves)
        else:
            avg_moves_per_ship = float('inf')  # Or use another indicator for "no ship sunk"

        return total_moves, avg_moves_per_ship

    def aggregate_runs(self, runs):
        """
        runs: list of get_results() dicts, one per variation
        → returns stats = { metric_key: { gens, mean, std } or per-baseline dict… }
        """
        agg = {}

        # --- Baseline‐dependent metrics: move_count & hit_to_sunk_ratio ---
        for key in ("move_count", "hit_to_sunk_ratio"):
            agg[key] = {}
            # each run[key] is baseline→{gen→val}
            for baseline in runs[0][key].keys():
                per_run = [r[key][baseline] for r in runs]
                gens, mean, std = self._collapse_variations(per_run)
                agg[key][baseline] = {"gens": gens, "mean": mean, "std": std}

        # --- Baseline‐independent metrics: sparsity, diversity, orientation ---
        for key in ("sparsity", "diversity", "orientation"):
            # runs[*][key] is {gen→val}
            per_run = [r[key] for r in runs]
            gens, mean, std = self._collapse_variations(per_run)
            agg[key] = {"gens": gens, "mean": mean, "std": std}

        return agg

    def _collapse_variations(self, list_of_gen2vals):
        """
        list_of_gen2vals: [ {gen: val, …}, … ] for each variation
        → sorted_gens, mean_array, std_array
        """
        all_gens = sorted({g for d in list_of_gen2vals for g in d}, key=self.gen_key)
        A = np.array([[d.get(g, 0.0) for g in all_gens]
                      for d in list_of_gen2vals])
        return all_gens, A.mean(axis=0), A.std(axis=0)

    def gen_key(self, x):
        """
        Extract generation number as int from keys like:
          - 'nn_gen20'
          - 'random'   (no digits → fallback to 0 or just return x if you prefer)
          - 5          (already an int)
        """
        if isinstance(x, str):
            m = re.search(r'(\d+)$', x)
            if m:
                return int(m.group(1))
            else:
                # no trailing digits → put these first or last as you like
                return -1
        return int(x)

    def plot_metrics(self):
        """Plot placement agent metrics over generations, including board heatmaps."""

        # === 1. Move Count and Hit-to-Sunk Ratio (baseline dependent) ===
        metrics = {
            "Move Count": self.move_count,
            "Hit-to-Sunk Ratio": self.hit_to_sunk_ratio,
        }

        for metric_name, metric_data in metrics.items():
            plt.figure(figsize=(10, 6))
            for baseline, values in metric_data.items():
                generations = sorted(values.keys())
                metric_values = [values[gen] for gen in generations]
                plt.plot(generations, metric_values, marker="o", label=baseline, linewidth=2)

            if generations:
                if len(generations) > 10:
                    step = max(1, len(generations) // 10)
                    plt.xticks(generations[::step], rotation=45)
                else:
                    plt.xticks(generations, rotation=45)

            plt.title(f"Placement Agent: {metric_name}")
            plt.xlabel("Generation")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === 2. Sparsity Over Generations (baseline independent) ===
        if self.sparsity:
            gens_sparse = sorted(self.sparsity.keys())
            sparsity_vals = [self.sparsity[g] for g in gens_sparse]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(gens_sparse, sparsity_vals, marker="o")
            ax.set_title("Average Sparsity Over Generations")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Sparsity (avg ship distance)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === 3. Diversity Over Generations ===
        if self.diversity:
            gens_div = sorted(self.diversity.keys())
            diversity_vals = [self.diversity[g] for g in gens_div]

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(gens_div, diversity_vals, marker="o")
            ax2.set_title("Population Diversity Over Generations")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Average Pairwise Chromosome Distance")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === 4. Orientation Percentage Over Generations ===
        if self.orientation:
            gens_orient = sorted(self.orientation.keys())
            orientation_vals = [self.orientation[g] for g in gens_orient]

            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.plot(gens_orient, orientation_vals, marker="o", color="blue", label="Vertical Orientation %")
            ax3.axhline(50, color="red", linestyle="--", label="50% Threshold")
            ax3.fill_between(
                gens_orient, orientation_vals, 50,
                where=(np.array(orientation_vals) >= 50),
                color="blue", alpha=0.2, interpolate=True, label="Vertical Superior"
            )
            ax3.fill_between(
                gens_orient, orientation_vals, 50,
                where=(np.array(orientation_vals) < 50),
                color="green", alpha=0.2, interpolate=True, label="Horizontal Superior"
            )
            ax3.set_title("Ship Orientation Over Generations\n(Above 50%: Vertical; Below 50%: Horizontal)")
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Vertical Orientation (%)")
            ax3.set_ylim(0, 100)
            ax3.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # === 5. Board Occupancy Heatmaps ===
        if self.boards_over_generations:
            generations = sorted(self.boards_over_generations.keys())
            boards = [self.boards_over_generations[g] for g in generations]

            # Only show a subset if too many
            max_shown = 20
            if len(boards) > max_shown:
                step = len(boards) // max_shown
                boards = boards[::step]
                generations = generations[::step]

            num_boards = len(boards)
            cols = 5
            rows = (num_boards + cols - 1) // cols  # ceil division

            fig4, ax4 = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            ax4 = ax4.flatten()

            im = None  # To keep track of the last image for colorbar

            for idx, (board, gen) in enumerate(zip(boards, generations)):
                im = ax4[idx].imshow(board, cmap="viridis", vmin=0, vmax=1)
                ax4[idx].set_title(f"Gen {gen}", pad=8)
                ax4[idx].set_xticks([])
                ax4[idx].set_yticks([])
                ax4[idx].axis("off")

            # Hide any unused subplots
            for j in range(idx + 1, len(ax4)):
                ax4[j].axis("off")

            # Add a single colorbar to the right of the figure
            cbar_ax = fig4.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            fig4.colorbar(im, cax=cbar_ax, label='Occupancy Frequency')

            fig4.suptitle("Ship Placement Frequency Over Generations", fontsize=14, y=1.02)
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Leave space on the right for colorbar
            plt.show()


    def plot_metrics_from_agg(self, stats, experiment: str):
        """
        stats: output of aggregate_runs()
        experiment: like "neat" or "rl"
        Plots placement metrics with mean±std fill‐between.
        """
        # === 1) Baseline‐dependent metrics AS SUBPLOTS ===
        for title, key in [("Move Count", "move_count"),
                           ("Hit-to-Sunk Ratio", "hit_to_sunk_ratio")]:
            # extract list of baselines
            baselines = list(stats[key].keys())
            n = len(baselines)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            for ax, baseline in zip(axes[0], baselines):
                data = stats[key][baseline]
                gens, m, s = data['gens'], data['mean'], data['std']
                x = np.arange(len(gens))

                ax.plot(x, m, 'o-', linewidth=2, markersize=6, label=baseline)
                ax.fill_between(x, m - s, m + s, alpha=0.2)
                ax.set_title(baseline)
                ax.set_xlabel("Gen")
                ax.set_xticks(x)
                ax.set_xticklabels([str(g) for g in gens], rotation=45)
                ax.grid(alpha=0.3)

            # only the first subplot gets the y‐label
            axes[0][0].set_ylabel(title)
            fig.suptitle(f"{experiment} — {title}")
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.show()

        # === 2) Baseline-independent metrics ===
        for title, key in [("Sparsity", "sparsity"),
                           ("Diversity", "diversity"),
                           ("Orientation % Vertical", "orientation")]:
            gens = stats[key]['gens']
            m, s = stats[key]['mean'], stats[key]['std']
            x = np.arange(len(gens))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x, m, "o-", linewidth=3, markersize=6)
            ax.fill_between(x, m - s, m + s, alpha=0.2)

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gens], rotation=45)
            ax.set_title(f"Placing {experiment} — {title}")
            ax.set_xlabel("Generation")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    def plot_combined_all(self, all_stats):
        """
        all_stats: dict mapping experiment_name -> aggregated_stats
                   (output of PlacementEvaluator.aggregate_runs)
        This will:
          - For move_count & hit_to_sunk_ratio: plot one mean±std curve PER EXPERIMENT PER BASELINE.
          - For sparsity, diversity, orientation: plot one mean±std curve PER EXPERIMENT.
        """

        # --- 1) Baseline‐dependent metrics ---
        # inside plot_combined_all, replace the bd‐metrics block with this
        bd_metrics = [
            ("Move Count", "move_count"),
            ("Hit-to-Sunk Ratio", "hit_to_sunk_ratio"),
        ]
        for title, key in bd_metrics:
            # assume all_stats has the same baselines under each experiment
            example = next(iter(all_stats.values()))
            baselines = list(example[key].keys())
            n = len(baselines)

            # no sharey → each subplot gets its own y‐axis
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

            # plot each experiment *on each* baseline
            for ax, baseline in zip(axes[0], baselines):
                mins, maxs = [], []
                for exp_label, stats in all_stats.items():
                    data = stats[key][baseline]
                    gens, m, s = data["gens"], data["mean"], data["std"]
                    x = np.arange(len(gens))

                    ax.plot(x, m, "o-", linewidth=2, label=exp_label)
                    ax.fill_between(x, m - s, m + s, alpha=0.2)

                    mins.append((m - s).min())
                    maxs.append((m + s).max())

                # tighten this subplot's y‐limit
                lo, hi = min(mins), max(maxs)
                pad = (hi - lo) * 0.1
                ax.set_ylim(lo - pad, hi + pad)

                # new per‐subplot title
                ax.set_title(f"Versus {baseline} baseline", fontsize=9)
                ax.set_xlabel("Generation", fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels([str(g) for g in gens], rotation=45)
                ax.grid(alpha=0.3)

            # only the first gets the y‐label
            axes[0][0].set_ylabel(title, fontsize=8)

            # overall figure title (metric name)
            fig.suptitle(f"Placement: {title} Across Experiments", fontsize=10)

            # collect handles & labels once
            seen = {}
            for ax in axes.flatten():
                for h, lab in zip(*ax.get_legend_handles_labels()):
                    if lab not in seen:
                        seen[lab] = h

            # 1) give the bottom of the figure more breathing room
            fig.subplots_adjust(bottom=0.3)  # increase bottom margin from 0.1 to 0.2

            # 2) place legend centered below all subplots
            fig.legend(
                seen.values(), seen.keys(),
                loc="lower center",  # use lower center
                ncol=len(seen),
                bbox_to_anchor=(0.5, -0.02),  # slightly below the axes
                bbox_transform=fig.transFigure,
                title="Opposing Agent"
            )

            plt.show()

        # --- 2) Baseline‐independent metrics ---
        bi_metrics = [
            ("Sparsity", "sparsity"),
            ("Diversity", "diversity"),
            ("Orientation % Vertical", "orientation"),
        ]
        for title, key in bi_metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            for exp_label, stats in all_stats.items():
                sk = stats[key]  # { 'gens':…, 'mean':…, 'std':… }
                gens = sk["gens"]
                m = sk["mean"]
                s = sk["std"]
                x = np.arange(len(gens))
                ax.plot(x, m, 'o-', linewidth=2, label=exp_label)
                ax.fill_between(x, m - s, m + s, alpha=0.2)

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gens], rotation=45)
            ax.set_title(f"Placement: {title}")
            ax.set_xlabel("Generation")
            ax.set_ylabel(title)
            ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(alpha=0.3)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    # ------------------- Helper to Create Board from Chromosome -------------------
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
    def compute_orientation(self, population):
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


class Evaluator:
    """
    Wrapper class that combines both search and placement evaluators.
    Maintains backward compatibility with existing code.
    """

    def __init__(self, game_manager, board_size, ship_sizes, num_evaluation_games=10):
        self.search_evaluator = SearchEvaluator(game_manager, board_size, ship_sizes, ["random", "uniform_spread"],
                                                num_evaluation_games, )
        self.placement_evaluator = PlacementEvaluator(game_manager, board_size, ship_sizes, ["random", "hunt_down"],
                                                      num_evaluation_games)

        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.num_evaluation_games = num_evaluation_games

    def evaluate_search_agents(self, search_agents, gen):
        """Proxy to the search evaluator."""
        for baseline_name in self.search_evaluator.baseline.keys():
            self.search_evaluator.evaluate(search_agents, gen, baseline_name)

    def evaluate_placement_agents(self, placement_agents, gen):
        """Proxy to the placement evaluator."""
        for baseline_name in self.placement_evaluator.baseline.keys():
            self.placement_evaluator.evaluate(placement_agents, gen, baseline_name)

    def plot_metrics_search(self):
        self.search_evaluator.plot_metrics()

    def plot_metrics_placement(self):
        self.placement_evaluator.plot_metrics()
