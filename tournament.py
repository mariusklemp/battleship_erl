import pygame

from game_logic.game import Game
from gui import GUI


def main(board_size=10, sizes=[5, 4, 3, 3, 2], human_player=False, player1_search_strategy="random",
         player2_search_strategy="random", player1_placing_strategy="random", player2_placing_strategy="random"):
    results = {1: [0, []], 2: [0, []]}

    for i in range(100):
        game = Game(board_size=board_size, sizes=sizes, player1_search_strategy=player1_search_strategy,
                    player2_search_strategy=player2_search_strategy,
                    player1_placing_strategy=player1_placing_strategy, player2_placing_strategy=player2_placing_strategy)

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
    main(board_size=10, player1_search_strategy="random", player1_placing_strategy="random",
         player2_search_strategy="hunt_down",  player2_placing_strategy="random")
