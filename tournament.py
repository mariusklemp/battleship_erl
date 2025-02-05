import pygame

from game_logic.game import Game
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from gui import GUI


def initialize_agents(board_size, sizes, search_strategy, placing_strategy):
    """Initializes Search and Placement Agents."""

    # model_100
    search_agent = SearchAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=search_strategy,
        net=None,
        optimizer=None,
        lr=None
    )
    placement_agent = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=placing_strategy,
    )
    return search_agent, placement_agent


def init_players(numb_players):
    # Create nets from saved models
    # return a dict of nets
    for i in range(numb_players):
        initialize_agents()
    return {}


def main(board_size, ship_sizes, placing_strategy, numb_players, numb_games=100):
    results = {1: [0, []], 2: [0, []]}

    players = init_players(numb_players)

    for i in range(numb_games):

        human_player = False

        while not game.game_over:
            game.play_turn() if human_player else game.play_turn()

        results[game.result][0] += 1
        results[game.result][1].append(
            game.player1Search.move_count if game.result == 1 else game.player2Search.move_count)

    print(f"Player 1 ({player1_search_strategy}) won {results[1][0]} times")
    print(f"Player 2 ({player2_search_strategy}) won {results[2][0]} times")

    print("Average move count for Player 1:", sum(results[1][1]) / len(results[1][1]))
    print("Average move count for Player 2:", sum(results[2][1]) / len(results[2][1]))


if __name__ == "__main__":
    main(board_size=5, placing_strategy="random", ship_sizes=[2, 1, 2], numb_players=2)
