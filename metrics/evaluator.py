from abc import abstractmethod, ABC

import numpy as np
import matplotlib.pyplot as plt

import visualize
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
            "Avg Moves to Win": np.mean(all_moves),
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

        # === Convert Grouped Bar Charts to Line Charts ===
        for group_title, meta in grouped_line_metrics.items():
            metric1, metric2 = meta["metrics"]
            color1, color2 = meta["colors"]

            data1 = getattr(self, metric1.lower().replace(" ", "_"))
            data2 = getattr(self, metric2.lower().replace(" ", "_"))

            avg_data1 = average_across_baselines(data1)
            avg_data2 = average_across_baselines(data2)

            all_generations = list(set(avg_data1.keys()) | set(avg_data2.keys()))
            all_generations = sorted(all_generations, key=lambda k: int(k))

            values1 = [avg_data1.get(gen, 0) for gen in all_generations]
            values2 = [avg_data2.get(gen, 0) for gen in all_generations]

            plt.figure(figsize=(12, 6))
            plt.plot(all_generations, values1, 'o-', label=metric1, color=color1, linewidth=2)
            plt.plot(all_generations, values2, 'o-', label=metric2, color=color2, linewidth=2)

            # Only show a subset of x-ticks if there are too many
            if len(all_generations) > 10:
                step = max(1, len(all_generations) // 10)
                plt.xticks(all_generations[::step], rotation=45)
            else:
                plt.xticks(all_generations, rotation=45)

            plt.title(f"Search Agent: {group_title}")
            plt.xlabel("Generation")
            plt.ylabel(group_title)
            plt.legend()
            plt.tight_layout()
            plt.grid(alpha=0.3)
            plt.show()

        # === Convert Individual Bar Charts to Line Charts ===
        for title, (attr_name, color) in single_line_metrics.items():
            data = getattr(self, attr_name)
            avg_data = average_across_baselines(data)

            all_generations = sorted(avg_data.keys(),
                                     key=lambda k: int(k.split("_")[1]) if isinstance(k, str) and "_" in k else int(k))
            values = [avg_data.get(gen, 0) for gen in all_generations]

            plt.figure(figsize=(10, 5))
            plt.plot(all_generations, values, 'o-', color=color, linewidth=2, label=title)

            # Only show a subset of x-ticks if there are too many
            if len(all_generations) > 10:
                step = max(1, len(all_generations) // 10)
                plt.xticks(all_generations[::step], rotation=45)
            else:
                plt.xticks(all_generations, rotation=45)

            plt.title(f"Search Agent: {title}")
            plt.xlabel("Generation")
            plt.ylabel(title)
            plt.tight_layout()
            plt.grid(alpha=0.3)
            plt.show()

        # === Line Plots ===
        for metric_name, metric_data in line_metrics.items():
            plt.figure(figsize=(10, 6))
            for baseline, values in metric_data.items():
                keys = sorted(list(values.keys()), key=lambda k: int(k))
                values_sorted = [values[k] for k in keys]
                plt.plot(keys, values_sorted, marker="o", label=baseline, linewidth=2)

            # Only show a subset of x-ticks if there are too many
            if len(keys) > 10:
                step = max(1, len(keys) // 10)
                plt.xticks(keys[::step], rotation=45)
            else:
                plt.xticks(keys, rotation=45)

            plt.title(f"Search Agent: {metric_name}")
            plt.xlabel("Generation")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

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
            "Avg Moves": normalize_avg_moves(m["Avg Moves to Win"]),
            "Hit Accuracy": normalize_hit_accuracy(m["Hit Accuracy"]),
            "Moves Between Hits": normalize_moves_between_hits(m["Moves Between Hits"]),
            "Sink Efficiency": normalize_sink_efficiency(m["Sink Efficiency"]),
        }
        return normalized_metrics

    def plot_final_skill_radar_chart(self, labeled_metrics, title="Skill of Final Iteration"):
        """
        Plot a radar chart for multiple agents' final evaluation.

        Args:
            labeled_metrics (list of tuples): Each is (label, metrics_dict).
            title (str): Chart title.
        """
        print("\n📊 Radar Chart Metrics:")
        for label, metrics in labeled_metrics:
            print(f"\n🔹 {label}:")
            for k, v in metrics.items():
                print(f"{k:<22}: {v:.3f}")

        # Define fixed metric order
        all_labels = ["Avg Moves", "Hit Accuracy", "Moves Between Hits", "Sink Efficiency"]
        angles = np.linspace(0, 2 * np.pi, len(all_labels), endpoint=False).tolist()
        angles += angles[:1]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))  # Wider figure
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_labels, fontsize=10)
        ax.set_yticklabels([])

        for label, metrics in labeled_metrics:
            normalized = self.normalize(metrics)
            values = [normalized[k] for k in all_labels] + [normalized[all_labels[0]]]

            # Use dotted lines for baselines
            is_baseline = any(x in label.lower() for x in ["random", "hunt", "mcts"])
            linestyle = "dotted" if is_baseline else "solid"
            alpha_fill = 0.08 if is_baseline else 0.15

            ax.plot(angles, values, linewidth=2, label=label, linestyle=linestyle)
            ax.fill(angles, values, alpha=alpha_fill)

        ax.set_title(title, size=14, pad=20)

        # Move legend outside the plot area to the right
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 0.5), fontsize=9, frameon=False)

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

    def plot_metrics(self):
        """Plot placement agent metrics (one figure per metric, one line per baseline)."""
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

            # Only show a subset of x-ticks if there are too many
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
