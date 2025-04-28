# metrics/competitive_evaluator.py
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

    def plot(self, hof=None, hos=None):
        """Plot the competitive evaluation metrics and optionally Hall of Fame and Hall of Shame boards."""
        generations = np.arange(len(self.placement_eval_history))

        # --- Combined Plot: Competitive Fitness Curves ---
        plt.figure(figsize=(10, 6))

        # Placement Fitness
        plt.plot(generations, self.placement_eval_history, marker="o", label="Placement Fitness", color="blue")
        if len(self.placement_eval_history) > 1:
            coeffs_place = np.polyfit(generations, self.placement_eval_history, 1)
            poly_place = np.poly1d(coeffs_place)
            plt.plot(generations, poly_place(generations), "b--", label="Placement Trend")

        # Search Fitness
        plt.plot(generations, self.search_eval_history, marker="o", label="Search Fitness", color="green")
        if len(self.search_eval_history) > 1:
            coeffs_search = np.polyfit(generations, self.search_eval_history, 1)
            poly_search = np.poly1d(coeffs_search)
            plt.plot(generations, poly_search(generations), "g--", label="Search Trend")

        plt.xlabel("Generation")
        plt.ylabel("Competitive Fitness")
        plt.title("Competitive Evaluation Metrics per Generation")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- NEW: Placement Fitness Only ---
        plt.figure(figsize=(8, 5))
        plt.plot(generations, self.placement_eval_history, marker="o", color="blue", label="Placement Fitness")
        if len(self.placement_eval_history) > 1:
            coeffs_place = np.polyfit(generations, self.placement_eval_history, 1)
            poly_place = np.poly1d(coeffs_place)
            plt.plot(generations, poly_place(generations), "b--", label="Placement Trend")
        plt.xlabel("Generation")
        plt.ylabel("Placement Fitness")
        plt.title("Placement Fitness over Generations")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- NEW: Search Fitness Only ---
        plt.figure(figsize=(8, 5))
        plt.plot(generations, self.search_eval_history, marker="o", color="green", label="Search Fitness")
        if len(self.search_eval_history) > 1:
            coeffs_search = np.polyfit(generations, self.search_eval_history, 1)
            poly_search = np.poly1d(coeffs_search)
            plt.plot(generations, poly_search(generations), "g--", label="Search Trend")
        plt.xlabel("Generation")
        plt.ylabel("Search Fitness")
        plt.title("Search Fitness over Generations")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Plot Hall of Fame ---
        if hof is not None and len(hof.items) > 0:
            num_hof = len(hof.items)
            fig, axs = plt.subplots(1, num_hof, figsize=(5 * num_hof, 5))
            if num_hof == 1:
                axs = [axs]
            for i, chromosome in enumerate(hof.items):
                board = self.create_board_from_chromosome(chromosome)
                axs[i].imshow(board, cmap="viridis", vmin=0, vmax=1)
                axs[i].set_title(f"HOF {i}")
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            plt.suptitle("Hall of Fame Boards")
            plt.tight_layout()
            plt.show()

        # --- Plot Hall of Shame ---
        if hos is not None and len(hos.items) > 0:
            num_hos = len(hos.items)
            fig2, axs2 = plt.subplots(1, num_hos, figsize=(5 * num_hos, 5))
            if num_hos == 1:
                axs2 = [axs2]
            for i, chromosome in enumerate(hos.items):
                board = self.create_board_from_chromosome(chromosome)
                axs2[i].imshow(board, cmap="viridis", vmin=0, vmax=1)
                axs2[i].set_title(f"HOS {i}")
                axs2[i].set_xticks([])
                axs2[i].set_yticks([])
            plt.suptitle("Hall of Shame Boards")
            plt.tight_layout()
            plt.show()

