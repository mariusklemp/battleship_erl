import pygame

from ai.mcts import MCTS
from game_logic.game import Game
from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from gui import GUI
from ai.models import ANET
import json


def initialize_agents(board_size, sizes, search_strategy, placing_strategy):
    """Initializes Search and Placement Agents."""
    layer_config = json.load(open("ai/config.json"))

    net = ANET(
        board_size=board_size,
        activation="relu",
        output_size=board_size**2,
        device="cpu",
        layer_config=layer_config,
        extra_input_size=6,
    )
    search_agent = SearchAgent(
        board_size=board_size,
        strategy=search_strategy,
        net=net,
        optimizer="adam",
        lr=0.001,
    )
    search_agent.strategy.load_model("models/model_100.pth")
    placement_agent = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=placing_strategy,
    )
    return search_agent, placement_agent


def initialize_game(
    board_size,
    sizes,
    human_player,
    player1_search_strategy,
    player1_placing_strategy,
    player2_search_strategy=None,
    player2_placing_strategy=None,
):
    """
    Initializes and runs a Battleship game.
    Supports both single-player (AI vs AI or Human vs AI) and two-player AI matches (Search 1 vs Placing 2).
    """

    # Player 1 (Search 1 vs Placing 2)
    search_agent_1, placement_agent_1 = initialize_agents(
        board_size, sizes, player1_search_strategy, player1_placing_strategy
    )

    # If player2_search_strategy is None, assume single-player mode
    if player2_search_strategy:
        search_agent_2, placement_agent_2 = initialize_agents(
            board_size, sizes, player2_search_strategy, player2_placing_strategy
        )
        game_manager_1 = GameManager(
            size=board_size, placing=placement_agent_2
        )  # Player 1 attacks Player 2’s board
        game_manager_2 = GameManager(
            size=board_size, placing=placement_agent_1
        )  # Player 2 attacks Player 1’s board
        game = Game(game_manager_1, search_agent_1, game_manager_2, search_agent_2)
        current_state_1 = game.game_manager1.initial_state()
        current_state_2 = game.game_manager2.initial_state()
    else:
        # Single-player mode (AI vs AI or Human vs AI)
        game_manager = GameManager(size=board_size, placing=placement_agent_1)
        game = Game(game_manager, search_agent_1)
        current_state_1 = game.game_manager1.initial_state()
        current_state_2 = None  # Not used in single-player mode

    # Initialize MCTS for AI players
    if player1_search_strategy == "mcts":
        mcts = MCTS(
            game_manager_1 if player2_search_strategy else game_manager,
            simulations_number=400,
            exploration_constant=1.41,
        )
        search_agent_1.strategy.set_mcts(mcts)

    if player2_search_strategy == "mcts":
        mcts = MCTS(game_manager_2, simulations_number=1000, exploration_constant=1.41)
        search_agent_2.strategy.set_mcts(mcts)

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Battleship")

    # Initialize GUI
    gui = GUI(board_size)
    gui.update_board(
        current_state_1, current_state_2 if player2_search_strategy else None
    )

    # Game Loop
    while not game.game_over:
        pygame.display.update()

        if human_player:
            current_state_1, current_state_2 = game.play_turn(
                gui=gui,
                current_state_1=current_state_1,
                current_state_2=current_state_2,
            )
        else:
            current_state_1, current_state_2 = game.play_turn(
                current_state_1, current_state_2
            )

        pygame.time.wait(200)
        gui.update_board(
            current_state_1, current_state_2 if player2_search_strategy else None
        )

    # Display result
    gui.display_win(game.winner, current_state_1.move_count)
    pygame.time.wait(1000)
    pygame.display.update()


if __name__ == "__main__":
    # Example: AI vs AI (Search 1 vs Placing 2, Search 2 vs Placing 1)
    initialize_game(
        board_size=5,
        sizes=[2, 1, 2],
        human_player=True,
        player1_search_strategy="nn_search",
        player1_placing_strategy="random",
        player2_search_strategy="nn_search",
        player2_placing_strategy="random",
    )

    # Example: Human vs AI
    # initialize_game(board_size=5, sizes=[2, 1, 2],
    #                 human_player=True,
    #                 player1_search_strategy="mcts",
    #                 player1_placing_strategy="random")
