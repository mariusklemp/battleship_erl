from game_logic.game_manager import Game


def main(board_size, sizes, search_strategy, placing_strategy):
    # Create Game instance and run the game
    game = Game(board_size=board_size, sizes=sizes, search_strategy=search_strategy, placing_strategy=placing_strategy)

    while not game.game_over:
        game.play_turn()

    print(f"Result: Used {game.move_count} moves!")


if __name__ == "__main__":
    main(board_size=10, sizes=[5, 4, 3, 3, 2], search_strategy="hunt_down", placing_strategy="random")
