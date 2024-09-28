import pygame

from game_logic.game_manager import Game
from gui import GUI


def main(board_size, sizes, human_player, player1_search_strategy, player1_placing_strategy, player2_search_strategy,player2_placing_strategy):
    # Create Game instance and run the game
    game = Game(board_size=board_size, sizes=sizes, player1_search_strategy=player1_search_strategy,player1_placing_strategy=player1_placing_strategy,
                player2_search_strategy=player2_search_strategy, player2_placing_strategy=player2_placing_strategy)

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Battleship")

    # Initialize board
    gui = GUI(board_size)
    gui.update_board(game)

    while not game.game_over:
        pygame.display.update()
        game.play_turn(gui=gui) if human_player else game.play_turn()
        pygame.time.wait(200)

        gui.update_board(game)

    gui.display_win(game.result)
    print(f"Result: Player {game.result} won!")
    if game.result == 1:
        print(f"The player used a {game.player1Search.strategy.name} strategy")
        print("Move count:", game.player1Search.move_count)
    else:
        print(f"The player used a {game.player2Search.strategy.name} strategy")
        print("Move count:", game.player2Search.move_count)
    pygame.time.wait(1000)

    pygame.display.update()


if __name__ == "__main__":
    main(board_size=10, sizes=[5, 4, 3, 3, 2], player1_search_strategy="random", player2_search_strategy="random",
         player1_placing_strategy="random", player2_placing_strategy="random", human_player=True)
