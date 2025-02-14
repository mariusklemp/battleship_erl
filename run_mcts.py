import pygame

from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from ai.mcts import MCTS
from game_logic.game_manager import GameManager
from tqdm import tqdm
from ai.models import ANET
from RBUF import RBUF
import json
import visualize
from gui import GUI
import torch
import numpy as np


def canonicalize_action_distribution(
    action_dist: np.ndarray, board_size: int, rotation: int
):
    """
    Rotate the action distribution (a flat NumPy array of length board_size^2)
    by the given number of 90° counterclockwise rotations.

    Args:
        action_dist: 1D NumPy array representing the probability distribution.
        board_size: int, the size of the board.
        rotation: int, number of 90° rotations applied during board canonicalization.

    Returns:
        A 1D NumPy array of the rotated (canonicalized) action distribution.
    """
    # Reshape to a 2D grid.
    dist_2d = action_dist.reshape(board_size, board_size)
    # Apply the rotation (np.rot90 rotates counterclockwise by default).
    canonical_dist_2d = np.rot90(dist_2d, k=rotation)
    # Flatten back to a 1D array.
    return canonical_dist_2d.flatten()


def simulate_game(
    game_manager, search_agent, mcts, rbuf, gui=None, placement_agent=None
):
    """Simulate a Battleship game and return the move count."""
    placement_agent.new_placements()
    current_state = game_manager.initial_state(
        placing=placement_agent
    )  # Reset the game and sets new board!
    if gui:
        gui.update_board(current_state)
        pygame.display.update()

    print("Ships to look for:")
    current_state.placing.show_ships()
    move_count = 0

    while not game_manager.is_terminal(current_state):
        if gui:
            gui.update_board(current_state)
            pygame.display.update()

        current_node = mcts.run(current_state, search_agent)
        best_child = current_node.best_child()

        move = best_child.move

        # Get the original board state and action distribution
        original_board = current_node.state.board
        original_action_dist = current_node.action_distribution(
            board_size=game_manager.size
        )

        # Assume state_tensor_canonical returns (canonical_board, extra_features, rotation)
        board_tensor, extra_features, rotation = (
            current_node.state.state_tensor_canonical()
        )

        # Retrieve the raw action distribution (a NumPy array).
        action_distribution = current_node.action_distribution(
            board_size=game_manager.size
        )

        # Check if we have a valid action distribution (not None)
        if action_distribution is not None:
            # Canonicalize the action distribution using the rotation value.
            canonical_action_distribution = canonicalize_action_distribution(
                action_distribution, board_size=game_manager.size, rotation=rotation
            )
        else:
            canonical_action_distribution = (
                action_distribution  # or handle appropriately
            )

        # Visualize original and canonicalized versions
        print("\nOriginal Board:")
        visualize.show_board(original_board, board_size=game_manager.size)

        print("\nOriginal Action Distribution:")
        visualize.plot_action_distribution(original_action_dist, game_manager.size)

        # Add the canonical state and action distribution to the replay buffer.
        rbuf.add_data_point(
            (
                (board_tensor, extra_features),  # The canonical input state.
                canonical_action_distribution,  # The canonicalized action distribution.
            )
        )

        current_state = game_manager.next_state(current_state, move)
        move_count += 1

    return move_count


def train_models(
    game_manager,
    mcts,
    rbuf,
    search_agent,
    number_actual_games,
    batch_size,
    M,
    device,
    graphic_visualiser,
    save_model,
    train_model,
    save_rbuf,
    board_size,
    placement_agent,
    play_game,
    epochs,
):
    move_count = []
    """Play a series of games, training and saving the model as specified."""
    gui = None
    # Initialize GUI
    if graphic_visualiser:
        pygame.init()
        pygame.display.set_caption("Battleship")
        gui = GUI(board_size)

    if save_model:
        search_agent.strategy.save_model(f"models/model_{0}.pth")

    for i in tqdm(range(number_actual_games)):
        if play_game:
            move_count.append(
                simulate_game(
                    game_manager,
                    search_agent,
                    mcts,
                    rbuf,
                    gui,
                    placement_agent,
                )
            )
            print(f"Finished game {i + 1} with {move_count[-1]} moves.")
            print("Replay buffer length:", len(rbuf.data))
            visualize.print_rbuf(rbuf, num_samples=5, board_size=game_manager.size)

        if train_model:
            batch = rbuf.get_training_set(batch_size)
            search_agent.strategy.train(batch, rbuf.get_validation_set(), device=device)
            search_agent.strategy.validate_model(rbuf.validation_set, device=device)


    if save_model and ((i + 1) % (number_actual_games / M) == 0):
            search_agent.strategy.save_model(f"models/model_{i + 1}.pth")

    if save_rbuf:
        rbuf.save_to_file(file_path="rbuf/rbuf.pkl")
    if train_model:
        search_agent.strategy.plot_metrics()

    if play_game:
        print(len(mcts.root_node.children))

    if play_game:
        visualize.plot_fitness(move_count, game_manager.size)


def main(
    board_size,
    sizes,
    strategy_placement,
    strategy_search,
    simulations_number,
    exploration_constant,
    M,
    epochs,
    number_actual_games,
    batch_size,
    device,
    load_rbuf,
    graphic_visualiser,
    save_model,
    train_model,
    save_rbuf,
    play_game,
):
    layer_config = json.load(open("ai/config.json"))

    net = ANET(
        board_size=board_size,
        activation="relu",
        output_size=board_size**2,
        device="cpu",
        layer_config=layer_config,
        extra_input_size=5,
    )

    search_agent = SearchAgent(
        board_size=board_size,
        strategy=strategy_search,
        net=net,
        optimizer="adam",
        lr=0.001,
    )
    placement_agent = PlacementAgent(
        board_size=board_size,
        ship_sizes=sizes,
        strategy=strategy_placement,
    )

    game_manager = GameManager(size=board_size)

    mcts = MCTS(
        game_manager,
        simulations_number=simulations_number,
        exploration_constant=exploration_constant,
    )

    rbuf = RBUF(max_len=10000)

    if load_rbuf:
        rbuf.load_from_file(file_path="rbuf/rbuf.pkl")
        print("Loaded replay buffer from file. Length:", len(rbuf.data))

    train_models(
        game_manager,
        mcts,
        rbuf,
        search_agent,
        number_actual_games,
        batch_size,
        M,
        device,
        graphic_visualiser,
        save_model,
        train_model,
        save_rbuf,
        board_size,
        placement_agent,
        play_game,
        epochs,
    )


if __name__ == "__main__":
    # Device setup with proper MPS handling
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        for _ in range(100):
            torch.matmul(
                torch.rand(500, 500).to(device), torch.rand(500, 500).to(device)
            )

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    main(
        board_size=5,
        sizes=[3, 2, 2],
        strategy_placement="random",
        strategy_search="nn_search",
        simulations_number=300,
        exploration_constant=1.41,
        M=10,
        epochs=20,
        number_actual_games=5000,
        batch_size=100,
        device=device,
        load_rbuf=True,
        graphic_visualiser=False,
        save_model=True,
        train_model=True,
        save_rbuf=True,
        play_game=False,
    )
