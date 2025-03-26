import json
import matplotlib.pyplot as plt
import numpy as np
import visualize
import math
from ai.mcts import MCTS
from ai.model import ANET
from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from tqdm import tqdm


class Tournament:
    """
    Class to manage the game state between players with different search strategies.
    Strategies may include "nn_search", "random", "hunt_down", "mcts", etc.
    """

    def __init__(
        self,
        board_size,
        num_games,
        ship_sizes,
        placing_strategy,
        search_strategies,
        num_players,
        game_manager,
        placement_agent,
    ):
        """
        Initialize the Tournament.

        :param board_size: Size of the board.
        :param num_games: Total number of games to play.
        :param ship_sizes: List of ship sizes.
        :param placing_strategy: The placing strategy (e.g. "random").
        :param search_strategies: A list of search strategy names (e.g. ["nn_search", "random", "hunt_down", "mcts"])
        """
        self.board_size = board_size
        self.num_games = num_games
        self.num_players = num_players
        self.ship_sizes = ship_sizes
        self.placing_strategy = placing_strategy
        self.search_strategies = search_strategies
        self.players = {}  # Dictionary mapping an identifier to a search agent
        
        # Store different metrics
        self.result = {}  # Dictionary mapping an identifier to a list of move counts
        self.hit_accuracy = {}  # Dictionary mapping an identifier to a list of hit accuracies
        self.sink_efficiency = {}  # Dictionary mapping an identifier to a list of moves between hit and sink
        self.moves_between_hits = {}  # Dictionary mapping an identifier to a list of avg moves between consecutive hits
        self.start_entropy = {}  # Dictionary mapping an identifier to a list of distribution entropy at start
        self.end_entropy = {}  # Dictionary mapping an identifier to a list of distribution entropy at end
        
        self.game_manager = game_manager
        self.placement_agent = placement_agent

    def set_nn_agent(self, i, layer_config):
        """Initializes a SearchAgent that uses a neural network.
        Model number is based on tournament parameters.
        """
        # Calculate model number to load models at regular intervals (0, 100, 200, ..., 1000)
        model_number = i * (self.num_games // self.num_players)
        print(f"[DEBUG] Loading model: {model_number}")
        path = f"models/model_{model_number}.pth"

        # Create a new network instance for each model
        net = ANET(
            board_size=self.board_size,
            activation="relu",
            device="cpu",
            layer_config=layer_config,
        )

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            name=f"nn_{model_number}",
            lr=0.001,
        )
        search_agent.strategy.load_model(path)
        return search_agent, f"nn_{model_number}"

    def init_players(self, time_limit):
        """
        Initialize players using the specified search strategies.
        For "nn_search", we load a neural network from file.
        For other strategies, we simply create a SearchAgent with that strategy.
        """
        # Load the network configuration
        layer_config = json.load(open("ai/config.json"))

        # For each strategy in the provided list, create a player.
        for i in range(0, self.num_players + 1):
            agent, identifier = self.set_nn_agent(i, layer_config)
            self.players[identifier] = agent
            self.result[identifier] = []  # Initialize an empty result list for this player
            self.hit_accuracy[identifier] = []  # Initialize hit accuracy tracking
            self.sink_efficiency[identifier] = []  # Initialize sink efficiency tracking
            self.moves_between_hits[identifier] = []  # Initialize moves between hits tracking
            self.start_entropy[identifier] = []  # Initialize start entropy tracking
            self.end_entropy[identifier] = []  # Initialize end entropy tracking

        for i, strat in enumerate(self.search_strategies):
            if strat in ["random", "hunt_down"]:
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strat,
                    name=strat,  # Add name for non-NN agents
                )
                identifier = f"{strat}"
            elif strat == "mcts":
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strat,
                    name=strat,  # Add name for MCTS agent
                    
                )
                mcts = MCTS(
                    self.game_manager, time_limit=time_limit, exploration_constant=1.41
                )
                agent.strategy.set_mcts(mcts)
                identifier = f"{strat}"
            else:
                raise ValueError(f"Unknown search strategy: {strat}")

            self.players[identifier] = agent
            self.result[identifier] = []  # Initialize an empty result list for this player
            self.hit_accuracy[identifier] = []  # Initialize hit accuracy tracking
            self.sink_efficiency[identifier] = []  # Initialize sink efficiency tracking
            self.moves_between_hits[identifier] = []  # Initialize moves between hits tracking
            self.start_entropy[identifier] = []  # Initialize start entropy tracking
            self.end_entropy[identifier] = []  # Initialize end entropy tracking
            
    def calculate_entropy(self, board):
        """
        Calculate the entropy of the distribution of shots on the board.
        Higher entropy means more uniform distribution (less concentrated).
        Lower entropy means more concentrated shooting pattern.
        
        :param board: A flattened board representation where 1s represent shots
        :return: The entropy value
        """
        # Get the board size
        flat_size = len(board)
        board_size = int(math.sqrt(flat_size))
        
        # Reshape the board for region analysis
        reshaped_board = np.array(board).reshape(board_size, board_size)
        
        # Divide the board into regions (e.g., 4x4 grid for a 10x10 board)
        region_size = max(2, board_size // 5)  # Ensure at least 2x2 regions
        n_regions = board_size // region_size
        
        # Count shots in each region
        region_counts = []
        for i in range(0, board_size, region_size):
            for j in range(0, board_size, region_size):
                i_end = min(i + region_size, board_size)
                j_end = min(j + region_size, board_size)
                region = reshaped_board[i:i_end, j:j_end]
                shot_count = np.sum(region)
                region_counts.append(shot_count)
        
        # Convert to probabilities
        total_shots = sum(region_counts)
        if total_shots == 0:  # No shots yet
            return 0
            
        probabilities = [count / total_shots for count in region_counts if count > 0]
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -sum(p * math.log(p, 2) for p in probabilities)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = math.log(len(region_counts), 2)
        if max_entropy == 0:
            return 0
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy

    def play(self, search_agent, game_manager):
        """Plays one game with the given search agent and returns the game metrics."""
        current_state = game_manager.initial_state(placing=self.placement_agent)
        
        # Metrics tracking
        total_moves = 0
        hits = 0
        misses = 0
        ship_hit_tracking = {}  # Track when each ship was first hit
        sink_moves = []  # Track moves between hitting and sinking ships
        
        # For tracking consecutive hits
        last_hit_move = None  # The move number of the last hit
        moves_between_hits_list = []  # List to store the number of moves between consecutive hits
        
        # For entropy calculation
        shots_made = []  # Track all shots at the end of the game
        start_shots = []  # Track the first 10 shots
        
        while not game_manager.is_terminal(current_state):
            total_moves += 1
            move = search_agent.strategy.find_move(current_state, topp=True)
            
            # Track shots for entropy calculation
            shots_made.append(move)
            if total_moves <= 10:  # Track first 10 shots for start entropy
                start_shots.append(move)
            
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
        
        # Calculate hit accuracy
        accuracy = hits / total_moves if total_moves > 0 else 0
        
        # Calculate average moves between hit and sink
        avg_sink_efficiency = sum(sink_moves) / len(sink_moves) if sink_moves else 0
        
        # Calculate average moves between consecutive hits
        avg_moves_between_hits = sum(moves_between_hits_list) / len(moves_between_hits_list) if moves_between_hits_list else 0
        
        # Calculate entropy metrics
        # Convert shot indices to a board representation for entropy calculation
        board_size_squared = self.board_size * self.board_size
        board_shots = [0] * board_size_squared
        for shot in shots_made:
            board_shots[shot] = 1
            
        # Calculate entropy for the full game
        end_entropy = self.calculate_entropy(board_shots)
        
        # Calculate entropy for the start of the game
        if start_shots:
            start_board = [0] * board_size_squared
            for shot in start_shots:
                start_board[shot] = 1
            start_entropy = self.calculate_entropy(start_board)
        else:
            start_entropy = 0
        
        return current_state.move_count, accuracy, avg_sink_efficiency, avg_moves_between_hits, start_entropy, end_entropy

    def plot_results(self):
        """
        Plots the tournament results with multiple metrics:
        1. Average move count per player
        2. Hit accuracy per player
        3. Sink efficiency (avg moves between hit and sink) per player
        4. Average moves between consecutive hits
        5. Start and end entropy
        """
        if not self.result:
            print("[DEBUG] No results to plot.")
            return

        # Sort identifiers with neural network models sorted numerically
        identifiers = list(self.result.keys())
        
        # Separate NN models from other strategies
        nn_models = [id for id in identifiers if id.startswith('nn_')]
        other_strategies = [id for id in identifiers if not id.startswith('nn_')]
        
        # Sort NN models numerically by extracting the number
        nn_models.sort(key=lambda x: int(x.split('_')[1]))
        
        # Sort other strategies alphabetically
        other_strategies.sort()
        
        # Combine the sorted lists
        identifiers = nn_models + other_strategies
        
        # Calculate averages for each metric
        avg_moves = []
        avg_accuracy = []
        avg_sink_efficiency = []
        avg_moves_btwn_hits = []
        avg_start_entropy = []
        avg_end_entropy = []
        
        for identifier in identifiers:
            # Move count
            moves = self.result[identifier]
            if moves:
                avg = sum(moves) / len(moves)
            else:
                avg = 0
            avg_moves.append(avg)
            
            # Hit accuracy
            accuracy = self.hit_accuracy[identifier]
            if accuracy:
                acc_avg = sum(accuracy) / len(accuracy)
            else:
                acc_avg = 0
            avg_accuracy.append(acc_avg)
            
            # Sink efficiency
            efficiency = self.sink_efficiency[identifier]
            if efficiency:
                eff_avg = sum(efficiency) / len(efficiency)
            else:
                eff_avg = 0
            avg_sink_efficiency.append(eff_avg)
            
            # Moves between hits
            btwn_hits = self.moves_between_hits[identifier]
            if btwn_hits:
                btwn_avg = sum(btwn_hits) / len(btwn_hits)
            else:
                btwn_avg = 0
            avg_moves_btwn_hits.append(btwn_avg)
            
            # Start entropy
            start_ent = self.start_entropy[identifier]
            if start_ent:
                start_ent_avg = sum(start_ent) / len(start_ent)
            else:
                start_ent_avg = 0
            avg_start_entropy.append(start_ent_avg)
            
            # End entropy
            end_ent = self.end_entropy[identifier]
            if end_ent:
                end_ent_avg = sum(end_ent) / len(end_ent)
            else:
                end_ent_avg = 0
            avg_end_entropy.append(end_ent_avg)
            
            print(
                f"[DEBUG] {identifier}: {len(moves)} games, "
                f"avg move count: {avg:.2f}, "
                f"hit accuracy: {acc_avg:.2f}, "
                f"sink efficiency: {eff_avg:.2f} moves, "
                f"moves between hits: {btwn_avg:.2f}, "
                f"start entropy: {start_ent_avg:.2f}, "
                f"end entropy: {end_ent_avg:.2f}"
            )

        # Create separate plots for each metric
        
        # Plot 1: Average Move Count
        plt.figure(figsize=(12, 6))
        bars = plt.bar(identifiers, avg_moves, color="skyblue")
        plt.xlabel("Player Identifier")
        plt.ylabel("Average Move Count")
        plt.title("Tournament Results: Average Move Count per Player")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig("plots/avg_move_count.png")
        plt.show()
        
        # Plot 2: Hit Accuracy
        plt.figure(figsize=(12, 6))
        bars = plt.bar(identifiers, avg_accuracy, color="lightgreen")
        plt.xlabel("Player Identifier")
        plt.ylabel("Hit Accuracy")
        plt.title("Tournament Results: Hit Accuracy per Player")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Accuracy is between 0 and 1
        plt.tight_layout()
        
        # Add values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig("plots/hit_accuracy.png")
        plt.show()
        
        # Plot 3: Sink Efficiency and Moves Between Hits
        plt.figure(figsize=(12, 6))
        x = np.arange(len(identifiers))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, avg_sink_efficiency, width, color="salmon", label="Moves to Sink")
        bars2 = plt.bar(x + width/2, avg_moves_btwn_hits, width, color="orange", label="Moves Between Hits")
        
        plt.xlabel("Player Identifier")
        plt.ylabel("Average Moves")
        plt.title("Tournament Results: Efficiency Metrics")
        plt.xticks(x, identifiers, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Add values on top of each bar
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig("plots/efficiency_metrics.png")
        plt.show()
        
        # Plot 4: Distribution Entropy
        plt.figure(figsize=(12, 6))
        x = np.arange(len(identifiers))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, avg_start_entropy, width, color="lightblue", label="Start Entropy")
        bars2 = plt.bar(x + width/2, avg_end_entropy, width, color="steelblue", label="End Entropy")
        
        plt.xlabel("Player Identifier")
        plt.ylabel("Entropy (0-1)")
        plt.title("Tournament Results: Distribution Entropy")
        plt.xticks(x, identifiers, rotation=45)
        plt.ylim(0, 1)  # Normalized entropy is between 0 and 1
        plt.legend()
        plt.tight_layout()
        
        # Add values on top of each bar
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig("plots/distribution_entropy.png")
        plt.show()
        
        # Plot 5: Combined performance score
        plt.figure(figsize=(12, 6))
        
        # Calculate combined performance score
        combined_score = []
        for i in range(len(identifiers)):
            # Lower is better for moves and moves between hits
            # Higher is better for accuracy
            # Normalize all to 0-1 range where 1 is best
            norm_moves = 1 - (avg_moves[i] / max(avg_moves)) if max(avg_moves) > 0 else 0
            norm_acc = avg_accuracy[i]
            norm_sink = 1 - (avg_sink_efficiency[i] / max(avg_sink_efficiency)) if max(avg_sink_efficiency) > 0 else 0
            norm_btwn = 1 - (avg_moves_btwn_hits[i] / max(avg_moves_btwn_hits)) if max(avg_moves_btwn_hits) > 0 else 0
            
            # Simple weighted sum (can be adjusted)
            score = (norm_moves + norm_acc + norm_sink + norm_btwn) / 4
            combined_score.append(score)
        
        bars = plt.bar(identifiers, combined_score, color="purple")
        plt.xlabel("Player Identifier")
        plt.ylabel("Combined Performance Score")
        plt.title("Tournament Results: Overall Performance (higher is better)")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Add values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig("plots/combined_score.png")
        plt.show()
        
        # Create directory for plots if it doesn't exist
        import os
        os.makedirs("plots", exist_ok=True)


def main(
    board_size,
    placing_strategy,
    ship_sizes,
    num_games,
    num_players,
    other_strategies=None,
    time_limit=0.5,
):

    placement_agent = PlacementAgent(
        board_size=board_size,
        ship_sizes=ship_sizes,
        strategy=placing_strategy,
    )
    game_manager = GameManager(size=board_size)

    tournament = Tournament(
        board_size,
        num_games,
        ship_sizes,
        placing_strategy,
        other_strategies,
        num_players,
        game_manager,
        placement_agent,
    )
    tournament.init_players(time_limit)

    for i in tqdm(range(int(num_games / 10)), desc="Tournament Progress"):
        placement_agent.new_placements()
        for identifier, search_agent in tournament.players.items():
            move_count, accuracy, sink_efficiency, moves_between_hits, start_entropy, end_entropy = tournament.play(search_agent, game_manager)
            tournament.result[identifier].append(move_count)
            tournament.hit_accuracy[identifier].append(accuracy)
            tournament.sink_efficiency[identifier].append(sink_efficiency)
            tournament.moves_between_hits[identifier].append(moves_between_hits)
            tournament.start_entropy[identifier].append(start_entropy)
            tournament.end_entropy[identifier].append(end_entropy)

    tournament.plot_results()

def load_config(config_path="config/mcts_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


if __name__ == "__main__":
    config = load_config()
    main(
        board_size=config["board_size"],
        placing_strategy="random",
        ship_sizes=config["ship_sizes"],
        num_games=config["training"]["number_actual_games"],
        num_players=10,
        other_strategies=["random", "hunt_down", "mcts"],
        time_limit=config["mcts"]["time_limit"],
    )
