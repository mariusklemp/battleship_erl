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
        Evaluate search agents against baseline placement agents.
        Runs multiple games and averages the results for more reliable metrics.
        """
        self.generations.append(generation)

        # Get the best search agent (assuming it's the last one in the list)
        best_search_agent = search_agents[-1]

        # Evaluate against baselines multiple times
        all_moves = []
        all_accuracy = []
        all_sink_efficiency = []
        all_moves_between_hits = []
        all_start_entropy = []
        all_end_entropy = []

        for _ in range(self.num_evaluation_games):
            # Create a new random placement agent for each game to ensure variety
            baseline = self.baseline[baseline_name]
            baseline.new_placements()

            # Simulate a game
            moves, accuracy, sink_efficiency, moves_between_hits, start_entropy, end_entropy = self.simulate_game(
                self.game_manager, baseline, best_search_agent
            )
            all_moves.append(moves)
            all_accuracy.append(accuracy)
            all_sink_efficiency.append(sink_efficiency)
            all_moves_between_hits.append(moves_between_hits)
            all_start_entropy.append(start_entropy)
            all_end_entropy.append(end_entropy)

        # Calculate averages
        avg_moves = np.mean(all_moves)
        avg_accuracy = np.mean(all_accuracy)
        avg_sink_efficiency = np.mean(all_sink_efficiency)
        avg_moves_between_hits = np.mean(all_moves_between_hits)
        avg_start_entropy = np.mean(all_start_entropy)
        avg_end_entropy = np.mean(all_end_entropy)

        # Store the average performance
        self.result[baseline_name][generation] = avg_moves
        self.hit_accuracy[baseline_name][generation] = avg_accuracy
        self.sink_efficiency[baseline_name][generation] = avg_sink_efficiency
        self.moves_between_hits[baseline_name][generation] = avg_moves_between_hits
        self.start_entropy[baseline_name][generation] = avg_start_entropy
        self.end_entropy[baseline_name][generation] = avg_end_entropy

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
        start_distributions = []
        end_distributions = []

        while not game_manager.is_terminal(current_state):

            move, probabilities_np = search_agent.strategy.find_move(current_state, topp=True)

            if total_moves <= 3:  # Track first 3 distributions for start entropy
                start_distributions.append(probabilities_np)
            elif total_moves >= current_state.move_count - 3:  # Track last 3 distributions for end entropy
                end_distributions.append(probabilities_np)

            # Check if move will be a hit before applying it
            is_hit = move in current_state.placing.indexes

            # Process the move
            new_state = game_manager.next_state(current_state, move)

            # Update hit/miss count
            if is_hit:
                hits += 1

                # Track moves between consecutive hits
                if last_hit_move is not None:
                    moves_between = total_moves - last_hit_move
                    moves_between_hits_list.append(moves_between)
                last_hit_move = total_moves

                # Check if this move sunk a ship
                sunk, ship_size, hit_ship = game_manager.check_ship_sunk(move, new_state.board, current_state.placing)

                # Track ship hits and sinking for calculating sink efficiency
                if hit_ship:
                    ship_id = tuple(sorted(hit_ship))  # Use the ship's indexes as identifier

                    # If this is the first hit on this ship, record the move number
                    if ship_id not in ship_hit_tracking:
                        ship_hit_tracking[ship_id] = total_moves

                    # If the ship was sunk with this move
                    if sunk:
                        # Calculate moves between first hit and sink
                        moves_to_sink = total_moves - ship_hit_tracking[ship_id] + 1
                        sink_moves.append(moves_to_sink)
            else:
                misses += 1

            current_state = new_state
            total_moves += 1

        # Calculate hit accuracy
        accuracy = hits / total_moves if total_moves > 0 else 0

        # Calculate average moves between hit and sink
        avg_sink_efficiency = sum(sink_moves) / len(sink_moves) if sink_moves else 0

        # Calculate average moves between consecutive hits
        avg_moves_between_hits = sum(moves_between_hits_list) / len(
            moves_between_hits_list) if moves_between_hits_list else 0

        # Calculate entropy metrics for action distributions
        # Average entropy of all distributions
        end_entropies = [self.calculate_entropy(dist) for dist in end_distributions]
        end_entropy = np.mean(end_entropies)

        # Average entropy of starting distributions
        start_entropies = [self.calculate_entropy(dist) for dist in start_distributions]
        start_entropy = np.mean(start_entropies)

        return current_state.move_count, accuracy, avg_sink_efficiency, avg_moves_between_hits, start_entropy, end_entropy

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
        """Plot grouped metrics as bar charts with one combined bar per generation/model."""

        group_bar_metrics = {
            "Distribution Entropy": {
                "metrics": ("Start Entropy", "End Entropy"),
                "colors": ("lightblue", "steelblue"),
            },
            "Efficiency Metrics": {
                "metrics": ("Sink Efficiency", "Moves Between Hits"),
                "colors": ("salmon", "orange"),
            },
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

        # Grouped bar charts with one bar per generation/model (averaged across baselines)
        for group_title, meta in group_bar_metrics.items():
            metric1, metric2 = meta["metrics"]
            color1, color2 = meta["colors"]

            data1 = getattr(self, metric1.lower().replace(" ", "_"))
            data2 = getattr(self, metric2.lower().replace(" ", "_"))

            avg_data1 = average_across_baselines(data1)
            avg_data2 = average_across_baselines(data2)

            all_generations = list(set(avg_data1.keys()) | set(avg_data2.keys()))
            try:
                all_generations = sorted(all_generations, key=lambda k: int(k.split("_")[1]) if "_" in k else int(k))
            except:
                all_generations = sorted(all_generations)

            x = np.arange(len(all_generations))
            values1 = [avg_data1.get(gen, 0) for gen in all_generations]
            values2 = [avg_data2.get(gen, 0) for gen in all_generations]

            width = 0.35
            plt.figure(figsize=(12, 6))
            bar1 = plt.bar(x - width / 2, values1, width, label=metric1, color=color1)
            bar2 = plt.bar(x + width / 2, values2, width, label=metric2, color=color2)

            for bars in [bar1, bar2]:
                for b in bars:
                    h = b.get_height()
                    plt.text(b.get_x() + b.get_width() / 2., h + 0.05, f'{h:.2f}',
                             ha='center', va='bottom', fontsize=8)

            plt.xticks(x, all_generations, rotation=45)
            plt.title(f"Search Agent: {group_title}")
            plt.xlabel("Generation or Model")
            plt.ylabel(group_title)
            plt.legend()
            plt.tight_layout()
            plt.grid(alpha=0.3)
            plt.show()

        # Line plots per baseline
        for metric_name, metric_data in line_metrics.items():
            plt.figure(figsize=(10, 6))
            for baseline, values in metric_data.items():
                keys = list(values.keys())
                values_sorted = [values[k] for k in keys]
                plt.plot(keys, values_sorted, marker="o", label=baseline)

            plt.title(f"Search Agent: {metric_name} over Generations")
            plt.xlabel("Generation or Model")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(alpha=0.3)
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

        # Get the best placement agent (assuming it's the first one in the list)
        best_placement_agent = placement_agents[0]  # Assuming the first one is the best

        # Evaluate against baseline search multiple times
        all_moves = []
        all_sparsity = []
        all_hit_to_sunk_ratio = []

        for _ in range(self.num_evaluation_games):
            moves, sparsity, hit_to_sunk_ratio = self.simulate_game(
                self.game_manager,
                best_placement_agent,
                self.baseline[baseline_name],
            )
            all_moves.append(moves)
            all_sparsity.append(sparsity)
            all_hit_to_sunk_ratio.append(hit_to_sunk_ratio)

        # Calculate averages for random search
        avg_moves = np.mean(all_moves)
        avg_sparsity = np.mean(all_sparsity)
        avg_hit_ratio = np.mean(all_hit_to_sunk_ratio)

        # Store the average performance
        self.move_count[baseline_name][generation] = avg_moves
        self.sparsity[baseline_name][generation] = avg_sparsity
        self.hit_to_sunk_ratio[baseline_name][generation] = avg_hit_ratio

    def simulate_game(self, game_manager, placement_agent, search_agent):
        """Simulate a game between a placement agent and search agent, returning placement metrics."""
        current_state = game_manager.initial_state(placing=placement_agent)

        total_moves = 0
        hits = 0
        misses = 0
        sunk_ships = 0

        # Measure sparsity: how spread out ships are
        all_ship_cells = current_state.placing.indexes
        xs, ys = zip(*[divmod(idx, self.board_size) for idx in all_ship_cells])
        x_range = max(xs) - min(xs) + 1
        y_range = max(ys) - min(ys) + 1
        sparsity = x_range * y_range / (self.board_size ** 2)

        while not game_manager.is_terminal(current_state):
            result = search_agent.strategy.find_move(current_state, topp=True)
            if isinstance(result, tuple):
                move, distribution = result
            else:
                move = result

            is_hit = move in current_state.placing.indexes
            next_state = game_manager.next_state(current_state, move)

            if is_hit:
                hits += 1
                sunk, size, ship_cells = game_manager.check_ship_sunk(move, next_state.board, current_state.placing)
                if sunk:
                    sunk_ships += 1
            else:
                misses += 1

            current_state = next_state
            total_moves += 1

        hit_to_sunk_ratio = sunk_ships / hits if hits > 0 else 0

        return total_moves, sparsity, hit_to_sunk_ratio

    def plot_metrics(self):
        """Plot placement agent metrics (one figure per metric, one line per baseline)."""
        metrics = {
            "Move Count": self.move_count,
            "Sparsity": self.sparsity,
            "Hit-to-Sunk Ratio": self.hit_to_sunk_ratio,
        }

        for metric_name, metric_data in metrics.items():
            plt.figure(figsize=(10, 6))
            for baseline, values in metric_data.items():
                generations = sorted(values.keys())
                metric_values = [values[gen] for gen in generations]
                plt.plot(generations, metric_values, marker="o", label=baseline)

            plt.title(f"Placement Agent: {metric_name} over Generations")
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
