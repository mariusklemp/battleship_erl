import numpy as np
import matplotlib.pyplot as plt


class SearchMetricsTracker:
    def __init__(self, search_agents):
        """Initialize metrics tracker for search agents."""
        self.agent_names = [agent.name for agent in search_agents]
        self.moves_per_agent = {name: [] for name in self.agent_names}
        self.hits_per_agent = {name: [] for name in self.agent_names}
        self.misses_per_agent = {name: [] for name in self.agent_names}
        self.games_played = {name: 0 for name in self.agent_names}

    def record_game(self, search_agent, moves, hits, misses):
        """Record metrics for a single game."""
        name = search_agent.name
        self.moves_per_agent[name].append(moves)
        self.hits_per_agent[name].append(hits)
        self.misses_per_agent[name].append(misses)
        self.games_played[name] += 1

    def update_agents(self, search_agents):
        """
        Update the list of agents being tracked.

        Args:
            search_agents: New list of search agents
        """
        # Get the names of the new agents
        new_agent_names = [agent.name for agent in search_agents]

        # Add any new agents to the tracking dictionaries
        for name in new_agent_names:
            if name not in self.agent_names:
                self.agent_names.append(name)
                self.moves_per_agent[name] = []
                self.hits_per_agent[name] = []
                self.misses_per_agent[name] = []
                self.games_played[name] = 0

    def plot_metrics(self):
        """Plot various metrics for search agents."""
        # Plot 1: Average moves per agent
        plt.figure(figsize=(10, 6))
        avg_moves = [np.mean(self.moves_per_agent[name]) for name in self.agent_names]
        std_moves = [np.std(self.moves_per_agent[name]) for name in self.agent_names]

        plt.bar(self.agent_names, avg_moves, yerr=std_moves, capsize=5)
        plt.title("Average Moves per Search Agent")
        plt.xlabel("Search Agent")
        plt.ylabel("Average Moves")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot 2: Hit/Miss ratio per agent
        plt.figure(figsize=(10, 6))
        hit_ratios = []
        for name in self.agent_names:
            total_hits = sum(self.hits_per_agent[name])
            total_attempts = total_hits + sum(self.misses_per_agent[name])
            ratio = total_hits / total_attempts if total_attempts > 0 else 0
            hit_ratios.append(ratio * 100)  # Convert to percentage

        plt.bar(self.agent_names, hit_ratios)
        plt.title("Hit Percentage per Search Agent")
        plt.xlabel("Search Agent")
        plt.ylabel("Hit Percentage (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot 3: Performance over time
        plt.figure(figsize=(10, 6))
        for name in self.agent_names:
            moves_array = np.array(self.moves_per_agent[name])
            # Calculate moving average if we have enough data points
            if len(moves_array) > 5:
                moving_avg = np.convolve(moves_array, np.ones(5) / 5, mode="valid")
                plt.plot(moving_avg, label=name)
            else:
                plt.plot(moves_array, label=name)

        plt.title("Search Agent Performance Over Time")
        plt.xlabel("Game Number")
        plt.ylabel("Moves (5-game moving average)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nSearch Agent Performance Summary:")
        print("-" * 50)
        for name in self.agent_names:
            avg_moves = np.mean(self.moves_per_agent[name])
            total_hits = sum(self.hits_per_agent[name])
            total_misses = sum(self.misses_per_agent[name])
            total_attempts = total_hits + total_misses
            if total_attempts > 0:
                hit_ratio = total_hits / total_attempts * 100
            else:
                hit_ratio = 0

            print(f"\nAgent: {name}")
            print(f"Games played: {self.games_played[name]}")
            print(f"Average moves per game: {avg_moves:.2f}")
            print(f"Hit ratio: {hit_ratio:.2f}%")
