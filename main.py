import visualize
from ai.mcts import MCTS
from game_logic.game import Game
from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from gui import GUI
from ai.model import ANET
import json

import pygame


def get_ship_cells(x, y, size, orientation):
    """Return a set of (col, row) tuples that the ship would occupy."""
    cells = set()
    if orientation == 0:  # horizontal
        for i in range(size):
            cells.add((x + i, y))
    else:  # vertical
        for i in range(size):
            cells.add((x, y + i))
    return cells


def is_valid_ship_placement(existing_placements, x, y, size, orientation, board_size, ship_sizes):
    """
    Check if a ship of given size and orientation at (x, y) is valid.
    existing_placements: list of tuples (x, y, orientation) already placed.
    ship_sizes: list of ship sizes corresponding to the order of placements.
    """
    new_cells = get_ship_cells(x, y, size, orientation)

    # Check boundaries
    for cell in new_cells:
        cx, cy = cell
        if cx < 0 or cx >= board_size or cy < 0 or cy >= board_size:
            return False

    # Check overlap with already placed ships
    for i, (px, py, porientation) in enumerate(existing_placements):
        placed_size = ship_sizes[i]  # Assuming placements follow the ship_sizes order
        placed_cells = get_ship_cells(px, py, placed_size, porientation)
        if new_cells & placed_cells:
            return False
    return True


def get_human_ship_placements_via_gui(gui, ship_sizes):
    """
    Let the user place ships manually via mouse clicks.
    Returns a list of tuples (x, y, direction) for each ship.
    """
    # Set GUI to placement mode (centered board)
    gui.set_mode("placement")

    placements = []
    current_orientation = 0  # 0: horizontal, 1: vertical
    current_ship_index = 0
    clock = pygame.time.Clock()

    # Prepare fonts for displaying text and title
    font = pygame.font.SysFont('Arial', 24)
    title_font = pygame.font.SysFont('Arial', 36)

    while current_ship_index < len(ship_sizes):
        gui.SCREEN.fill((40, 50, 60))

        # Compute centered board offset
        board_left = (gui.WIDTH - gui.BOARD_SIZE * gui.SQUARE_SIZE) // 2
        board_top = (gui.HEIGHT - gui.BOARD_SIZE * gui.SQUARE_SIZE) // 2

        gui.draw_grid(state=None, left=board_left, top=board_top)

        # Draw a title at the top
        title_text = "Place Your Ships"
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(gui.WIDTH // 2, 30))
        gui.SCREEN.blit(title_surface, title_rect)

        # Draw already placed ships relative to centered board
        for i, (x, y, orientation) in enumerate(placements):
            size = ship_sizes[i]
            rect_x = board_left + x * gui.SQUARE_SIZE + gui.INDENT
            rect_y = board_top + y * gui.SQUARE_SIZE + gui.INDENT
            if orientation == 0:  # horizontal
                width = size * gui.SQUARE_SIZE - 2 * gui.INDENT
                height = gui.SQUARE_SIZE - 2 * gui.INDENT
            else:  # vertical
                width = gui.SQUARE_SIZE - 2 * gui.INDENT
                height = size * gui.SQUARE_SIZE - 2 * gui.INDENT
            ship_rect = pygame.Rect(rect_x, rect_y, width, height)
            pygame.draw.rect(gui.SCREEN, (50, 200, 150), ship_rect, border_radius=15)

        # Display information about the current ship
        current_ship_size = ship_sizes[current_ship_index]
        orientation_text = "Horizontal" if current_orientation == 0 else "Vertical"
        info_text = f"Placing ship {current_ship_index + 1}/{len(ship_sizes)}: Size {current_ship_size} ({orientation_text})"
        text_surface = font.render(info_text, True, (255, 255, 255))
        gui.SCREEN.blit(text_surface, (10, gui.HEIGHT - 40))

        # Display instruction to toggle orientation
        instruction_text = "Press SPACE to toggle orientation"
        inst_surface = font.render(instruction_text, True, (200, 200, 200))
        gui.SCREEN.blit(inst_surface, (10, gui.HEIGHT - 70))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_orientation = 1 - current_orientation

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Convert the click position to board coordinates based on the centered board
                col = (pos[0] - board_left) // gui.SQUARE_SIZE
                row = (pos[1] - board_top) // gui.SQUARE_SIZE

                if is_valid_ship_placement(placements, col, row, current_ship_size, current_orientation, gui.BOARD_SIZE,
                                           ship_sizes):
                    placements.append((col, row, current_orientation))
                    current_ship_index += 1
                else:
                    print("Invalid placement, try again.")

        clock.tick(30)

    # Switch to game layout after placement is complete
    gui.set_mode("game")
    return placements


def initialize_agents(board_size, sizes, search_strategy, placing_strategy, file_path, chromosome=None):
    """Initializes Search and Placement Agents."""

    net = ANET(
        board_size=board_size,
        activation="relu",
        device="cpu",
        layer_config="ai/config_simple.json",
    )
    search_agent = SearchAgent(
        board_size=board_size,
        strategy=search_strategy,
        net=net,
        optimizer="adam",
        lr=0.001,
    )

    if file_path:
        search_agent.strategy.net.load_model(file_path)

    placement_agent = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=placing_strategy,
        chromosome=chromosome,
    )
    return search_agent, placement_agent


def initialize_game(
        board_size,
        sizes,
        human_player,
        file_path_1,
        file_path_2,
        player1_search_strategy,
        player1_placing_strategy,
        player2_search_strategy=None,
        player2_placing_strategy=None,
):
    """
    Initializes and runs a Battleship game.
    Supports both single-player (AI vs AI or Human vs AI) and two-player AI matches (Search 1 vs Placing 2).
    """

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Battleship")

    # Initialize GUI
    gui = GUI(board_size, human_player)

    search_agent_1, placement_agent_1 = initialize_agents(
        board_size, sizes, player1_search_strategy, player1_placing_strategy, file_path_1
    )

    # If player2_search_strategy is None, assume single-player mode
    if player2_search_strategy:

        if human_player:
            chromosome = get_human_ship_placements_via_gui(gui, sizes)
            search_agent_2, placement_agent_2 = initialize_agents(
                board_size, sizes, player2_search_strategy, player2_placing_strategy, file_path_2, chromosome,
            )
        else:
            search_agent_2, placement_agent_2 = initialize_agents(
                board_size, sizes, player2_search_strategy, player2_placing_strategy, file_path_2
            )
        game_manager_1 = GameManager(
            size=board_size
        )  # Player 1 attacks Player 2’s board
        game_manager_2 = GameManager(
            size=board_size
        )  # Player 2 attacks Player 1’s board
        game = Game(game_manager_1, search_agent_1, game_manager_2, search_agent_2)
        current_state_1 = game.game_manager1.initial_state(placement_agent_1)
        current_state_2 = game.game_manager2.initial_state(placement_agent_2)
    else:
        # Single-player mode (AI vs AI or Human vs AI)
        game_manager = GameManager(size=board_size)
        game = Game(game_manager, search_agent_1)
        current_state_1 = game.game_manager1.initial_state(placement_agent_1)
        current_state_2 = None  # Not used in single-player mode

    # Initialize MCTS for AI players
    if player1_search_strategy == "mcts":
        mcts = MCTS(
            game_manager_1 if player2_search_strategy else game_manager,
            simulations_number=1000,
            exploration_constant=1.41,
        )
        search_agent_1.strategy.set_mcts(mcts)

    if player2_search_strategy == "mcts":
        mcts = MCTS(game_manager_2, simulations_number=10000, exploration_constant=1.41)
        search_agent_2.strategy.set_mcts(mcts)

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
    board_size = 5
    board = [[1, 0, 1, 1, 0,
              1, 0, 1, 0, 1,
              1, 0, 1, 0, 0,
              0, 1, 1, 0, 1,
              1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0,
              1, 0, 0, 0, 0,
              1, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1,
              0, 0, 0, 0, 0,
              0, 1, 0, 1, 0,
              0, 0, 0, 1, 0,
              1, 0, 1, 1, 0],
             [1, 1, 1, 0, 0,
              1, 0, 0, 0, 0,
              1, 0, 0, 0, 1,
              0, 0, 0, 0, 1,
              0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0]  # Example board, replace with actual board data
             ]
    visualize.show_board(board, board_size)
    initialize_game(
        board_size=5,
        sizes=[3, 3, 2],
        human_player=True,
        file_path_1="models/5/rl/solo/8/model_gen100.pth",
        file_path_2="models/5/rl/solo/8/model_gen100.pth",
        player1_search_strategy="nn_search",
        player1_placing_strategy="random",
        player2_search_strategy="nn_search",
        player2_placing_strategy="chromosome",
    )

    # Example: Human vs AI
    # initialize_game(board_size=5, sizes=[3, 2, 2],
    #                human_player=True,
    #                player1_search_strategy="nn_search",
    #                player1_placing_strategy="random",
    #                file_path_1="models/model_100.pth",
    #                file_path_2=""
    #                )
