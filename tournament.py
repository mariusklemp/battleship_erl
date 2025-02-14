import json
import matplotlib.pyplot as plt
import visualize
from ai.mcts import MCTS
from ai.models import ANET
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
        self.result = {}  # Dictionary mapping an identifier to a list of move counts
        self.game_manager = game_manager
        self.placement_agent = placement_agent

    def set_nn_agent(self, i, default_net):
        """Initializes a SearchAgent that uses a neural network.
        Model number is based on tournament parameters.
        """
        model_number = int(self.num_games / self.num_players * (i))
        print(f"[DEBUG] Loading model: {model_number}")
        path = f"models/model_{model_number}.pth"

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=default_net,
            optimizer="adam",
            name=f"nn_{model_number}",
            lr=0.001,
        )
        search_agent.strategy.load_model(path)
        return search_agent, f"nn_{model_number}"

    def init_players(self):
        """
        Initialize players using the specified search strategies.
        For "nn_search", we load a neural network from file.
        For other strategies, we simply create a SearchAgent with that strategy.
        """
        # Load the default network for NN-based players.
        layer_config = json.load(open("ai/config.json"))
        default_net = ANET(
            board_size=self.board_size,
            activation="relu",
            output_size=self.board_size**2,
            device="cpu",
            layer_config=layer_config,
        )

        # For each strategy in the provided list, create a player.
        for i in range(0, self.num_players + 1):
            agent, identifier = self.set_nn_agent(i, default_net)
            self.players[identifier] = agent
            self.result[identifier] = (
                []
            )  # Initialize an empty result list for this player

        for i, strat in enumerate(self.search_strategies):
            if strat in ["random", "hunt_down"]:
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strat,
                )

                identifier = f"{strat}"
            elif strat == "mcts":
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strat,
                )
                mcts = MCTS(
                    self.game_manager, simulations_number=500, exploration_constant=1.41
                )
                agent.strategy.set_mcts(mcts)

                identifier = f"{strat}"
            else:
                raise ValueError(f"Unknown search strategy: {strat}")

            self.players[identifier] = agent
            self.result[identifier] = (
                []
            )  # Initialize an empty result list for this player

    def play(self, search_agent, game_manager):
        """Plays one game with the given search agent and returns the move count."""
        current_state = game_manager.initial_state(placing=self.placement_agent)
        while not game_manager.is_terminal(current_state):
            visualize.show_board(current_state.board, board_size=game_manager.size)

            move = search_agent.strategy.find_move(current_state, topp=True)
            current_state = game_manager.next_state(current_state, move)
        return current_state.move_count

    def plot_results(self):
        """
        Plots the tournament results as a bar chart.
        The x-axis shows each player (by identifier) and the y-axis shows the average move count.
        Lower average move counts are considered better.
        """
        if not self.result:
            print("[DEBUG] No results to plot.")
            return

        identifiers = sorted(self.result.keys())
        avg_moves = []
        for identifier in identifiers:
            moves = self.result[identifier]
            if moves:
                avg = sum(moves) / len(moves)
            else:
                avg = 0
            avg_moves.append(avg)
            print(
                f"[DEBUG] {identifier}: {len(moves)} games, average move count: {avg:.2f}"
            )

        plt.figure(figsize=(10, 6))
        plt.bar(identifiers, avg_moves, color="skyblue")
        plt.xlabel("Player Identifier")
        plt.ylabel("Average Move Count")
        plt.title("Tournament Results: Average Move Count per Player")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main(
    board_size,
    placing_strategy,
    ship_sizes,
    num_games,
    num_players,
    other_strategies=None,
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
    tournament.init_players()

    for i in tqdm(range(int(num_games / 10)), desc="Tournament Progress"):
        placement_agent.new_placements()
        for identifier, search_agent in tournament.players.items():
            move_count = tournament.play(search_agent, game_manager)
            tournament.result[identifier].append(move_count)

    tournament.plot_results()


if __name__ == "__main__":
    main(
        board_size=5,
        placing_strategy="random",
        ship_sizes=[3, 2, 2],
        num_games=1000,
        num_players=10,
        other_strategies=["random", "hunt_down", "mcts"],
    )
