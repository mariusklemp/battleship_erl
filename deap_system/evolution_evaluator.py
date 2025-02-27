import numpy as np
import matplotlib.pyplot as plt
from game_logic.search_agent import SearchAgent
from game_logic.placement_agent import PlacementAgent
from game_logic.game_manager import GameManager


class EvolutionEvaluator:
    """
    Class to evaluate the performance of evolved agents against baseline opponents.
    Tracks performance metrics over generations to assess if competitive co-evolution
    is improving both search and placement agents.
    """

    def __init__(self, board_size, ship_sizes, num_evaluation_games=10):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.game_manager = GameManager(size=board_size)
        self.num_evaluation_games = num_evaluation_games

        # Create baseline opponents
        self.baseline_search_agents = {
            "random": SearchAgent(
                board_size=board_size, strategy="random", name="random_baseline"
            ),
            "hunt_down": SearchAgent(
                board_size=board_size, strategy="hunt_down", name="hunt_down_baseline"
            ),
        }

        self.baseline_placement_agents = {
            "random": PlacementAgent(
                board_size=board_size, ship_sizes=ship_sizes, strategy="random"
            )
        }

        # Metrics for tracking performance over generations
        self.search_performance = {
            "random": [],  # Average moves against random placement
        }

        self.placement_performance = {
            "random": [],  # Average moves by random search agent
            "hunt_down": [],  # Average moves by hunt_down search agent
        }

        # Generation numbers for plotting
        self.generations = []

    def simulate_game(self, game_manager, placing_agent, search_agent):
        """Simulate a game between a placing agent and a search agent."""
        state = game_manager.initial_state(placing_agent)
        moves = 0
        hits = 0
        misses = 0

        while not game_manager.is_terminal(state):
            move = search_agent.strategy.find_move(state)
            next_state = game_manager.next_state(state, move)

            # Track hits and misses
            if next_state.board[1][move] == 1:  # Hit
                hits += 1
            else:  # Miss
                misses += 1

            state = next_state
            moves += 1

        return moves, hits, misses

    def evaluate_search_agents(self, search_agents, generation):
        """
        Evaluate search agents against baseline placement agents.
        Runs multiple games and averages the results for more reliable metrics.
        """
        self.generations.append(generation)

        # Get the best search agent (assuming it's the last one in the list)
        best_search_agent = search_agents[-1]

        # Evaluate against random placement multiple times
        all_moves = []
        all_hits = []
        all_misses = []

        for _ in range(self.num_evaluation_games):
            # Create a new random placement agent for each game to ensure variety
            random_placement = PlacementAgent(
                board_size=self.board_size,
                ship_sizes=self.ship_sizes,
                strategy="random",
            )

            moves, hits, misses = self.simulate_game(
                self.game_manager, random_placement, best_search_agent
            )
            all_moves.append(moves)
            all_hits.append(hits)
            all_misses.append(misses)

        # Calculate averages
        avg_moves = np.mean(all_moves)
        avg_hits = np.mean(all_hits)
        avg_misses = np.mean(all_misses)
        hit_ratio = sum(all_hits) / (sum(all_hits) + sum(all_misses))

        # Store the average performance
        self.search_performance["random"].append(avg_moves)

        print(f"\nSearch Agent Evaluation (Generation {generation}):")
        print(f"  - Against random placement ({self.num_evaluation_games} games):")
        print(
            f"    - Average moves: {avg_moves:.2f} (min: {min(all_moves)}, max: {max(all_moves)})"
        )
        print(f"    - Average hits: {avg_hits:.2f}, Average misses: {avg_misses:.2f}")
        print(f"    - Hit ratio: {hit_ratio:.2f}")

    def evaluate_placement_agents(self, placement_agents, generation):
        """
        Evaluate placement agents against baseline search agents.
        Runs multiple games and averages the results for more reliable metrics.
        """
        # Get the best placement agent (assuming it's the first one in the list)
        best_placement_agent = placement_agents[0]  # Assuming the first one is the best

        # Evaluate against random search multiple times
        random_moves = []
        random_hits = []
        random_misses = []

        for _ in range(self.num_evaluation_games):
            moves, hits, misses = self.simulate_game(
                self.game_manager,
                best_placement_agent,
                self.baseline_search_agents["random"],
            )
            random_moves.append(moves)
            random_hits.append(hits)
            random_misses.append(misses)

        # Calculate averages for random search
        avg_random_moves = np.mean(random_moves)
        avg_random_hits = np.mean(random_hits)
        avg_random_misses = np.mean(random_misses)
        random_hit_ratio = sum(random_hits) / (sum(random_hits) + sum(random_misses))

        # Store the average performance
        self.placement_performance["random"].append(avg_random_moves)

        # Evaluate against hunt_down search multiple times
        huntdown_moves = []
        huntdown_hits = []
        huntdown_misses = []

        for _ in range(self.num_evaluation_games):
            moves, hits, misses = self.simulate_game(
                self.game_manager,
                best_placement_agent,
                self.baseline_search_agents["hunt_down"],
            )
            huntdown_moves.append(moves)
            huntdown_hits.append(hits)
            huntdown_misses.append(misses)

        # Calculate averages for hunt_down search
        avg_huntdown_moves = np.mean(huntdown_moves)
        avg_huntdown_hits = np.mean(huntdown_hits)
        avg_huntdown_misses = np.mean(huntdown_misses)
        huntdown_hit_ratio = sum(huntdown_hits) / (
            sum(huntdown_hits) + sum(huntdown_misses)
        )

        # Store the average performance
        self.placement_performance["hunt_down"].append(avg_huntdown_moves)

        print(f"\nPlacement Agent Evaluation (Generation {generation}):")
        print(f"  - Against random search ({self.num_evaluation_games} games):")
        print(
            f"    - Average moves: {avg_random_moves:.2f} (min: {min(random_moves)}, max: {max(random_moves)})"
        )
        print(
            f"    - Average hits: {avg_random_hits:.2f}, Average misses: {avg_random_misses:.2f}"
        )
        print(f"    - Hit ratio: {random_hit_ratio:.2f}")

        print(f"  - Against hunt_down search ({self.num_evaluation_games} games):")
        print(
            f"    - Average moves: {avg_huntdown_moves:.2f} (min: {min(huntdown_moves)}, max: {max(huntdown_moves)})"
        )
        print(
            f"    - Average hits: {avg_huntdown_hits:.2f}, Average misses: {avg_huntdown_misses:.2f}"
        )
        print(f"    - Hit ratio: {huntdown_hit_ratio:.2f}")

    def plot_metrics(self):
        """Plot performance metrics over generations."""
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Search Agent Performance
        ax1.plot(
            self.generations,
            self.search_performance["random"],
            "o-",
            label="vs Random Placement",
        )

        # Add trend lines
        if len(self.generations) > 1:
            # Random placement trend
            z = np.polyfit(self.generations, self.search_performance["random"], 1)
            p = np.poly1d(z)
            ax1.plot(self.generations, p(self.generations), "r--", alpha=0.5)

        ax1.set_title(
            f"Search Agent Performance Over Generations\n({self.num_evaluation_games} games per evaluation)"
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Average Moves (lower is better)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Placement Agent Performance
        ax2.plot(
            self.generations,
            self.placement_performance["random"],
            "o-",
            label="vs Random Search",
        )
        ax2.plot(
            self.generations,
            self.placement_performance["hunt_down"],
            "o-",
            label="vs Hunt Down Search",
        )

        # Add trend lines
        if len(self.generations) > 1:
            # Random search trend
            z = np.polyfit(self.generations, self.placement_performance["random"], 1)
            p = np.poly1d(z)
            ax2.plot(self.generations, p(self.generations), "r--", alpha=0.5)

            # Hunt down search trend
            z = np.polyfit(self.generations, self.placement_performance["hunt_down"], 1)
            p = np.poly1d(z)
            ax2.plot(self.generations, p(self.generations), "r--", alpha=0.5)

        ax2.set_title(
            f"Placement Agent Performance Over Generations\n({self.num_evaluation_games} games per evaluation)"
        )
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Average Moves (higher is better)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("evolution_performance.png")
        plt.show()

        # Print summary statistics
        print("\nEvolution Performance Summary:")
        print("-" * 50)

        # Search agent improvement
        if len(self.generations) > 1:
            search_random_improvement = (
                (
                    self.search_performance["random"][0]
                    - self.search_performance["random"][-1]
                )
                / self.search_performance["random"][0]
                * 100
            )

            print(f"\nSearch Agent Improvement:")
            print(
                f"  - Against random placement: {search_random_improvement:.2f}% improvement"
            )

            # Placement agent improvement
            placement_random_improvement = (
                (
                    self.placement_performance["random"][-1]
                    - self.placement_performance["random"][0]
                )
                / self.placement_performance["random"][0]
                * 100
            )
            placement_hunt_down_improvement = (
                (
                    self.placement_performance["hunt_down"][-1]
                    - self.placement_performance["hunt_down"][0]
                )
                / self.placement_performance["hunt_down"][0]
                * 100
            )

            print(f"\nPlacement Agent Improvement:")
            print(
                f"  - Against random search: {placement_random_improvement:.2f}% improvement"
            )
            print(
                f"  - Against hunt_down search: {placement_hunt_down_improvement:.2f}% improvement"
            )
