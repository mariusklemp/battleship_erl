import pygame

from ai.mcts import MCTS
from game_logic.game import Game
from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from gui import GUI


def main(board_size, sizes, human_player, player1_search_strategy, player1_placing_strategy, player2_search_strategy,
         player2_placing_strategy):

    # Create agents
    search_agent_1 = SearchAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=player1_search_strategy,
        net=None,
        optimizer=None,
        lr=None
    )
    placement_agent_1 = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=player1_placing_strategy,
    )

    search_agent_2 = SearchAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=player2_search_strategy,
        net=None,
        optimizer=None,
        lr=None,
    )
    placement_agent_2 = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=player2_placing_strategy,
    )

    game_manager_1 = GameManager(size=board_size, placing=placement_agent_2)
    game_manager_2 = GameManager(size=board_size, placing=placement_agent_1)

    mcts = None
    if player2_search_strategy == "mcts":
        mcts = MCTS(
            game_manager_1,
            simulations_number=400,
            exploration_constant=1.41,
        )
        search_agent_2.strategy.set_mcts(mcts)

    # Create Game instance and run the game
    game = Game(game_manager_1, game_manager_2, search_agent_1, search_agent_2)

    current_state_1 = game.game_manager1.initial_state()
    current_state_2 = game.game_manager2.initial_state()


    print("Game manager 1 search 1 vs placing 2")
    game_manager_1.placing.show_ships()

    print("Game manager 2 search 2 vs placing 1")
    game_manager_2.placing.show_ships()

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Battleship")

    # Initialize board
    gui = GUI(board_size)
    gui.update_board(current_state_1, current_state_2)

    while not game.game_over:
        pygame.display.update()

        current_state_1, current_state_2 = game.play_turn(gui=gui, current_state_1=current_state_1,
                                                          current_state_2=current_state_2) if human_player else game.play_turn(
                                                                                                current_state_1, current_state_2)

        pygame.time.wait(200)
        gui.update_board(current_state_1, current_state_2)

    gui.display_win(game.winner)
    print(f"Result: Player {game.winner} won!")
    pygame.time.wait(1000)

    pygame.display.update()


if __name__ == "__main__":
    main(board_size=3, sizes=[2, 1, 2],
         human_player=True,
         player1_search_strategy="mcts",
         player1_placing_strategy="random",
         player2_search_strategy="mcts",
         player2_placing_strategy="random",
        )
