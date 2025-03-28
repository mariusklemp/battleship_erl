import pygame
import json
import torch
import numpy as np
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from ai.mcts import MCTS
from game_logic.game_manager import GameManager
from tqdm import tqdm
from ai.model import ANET
from RBUF import RBUF
import visualize
from gui import GUI


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
    mcts.root_node = None
    if gui:
        gui.update_board(current_state)
        pygame.display.update()

    move_count = 0

    while not game_manager.is_terminal(current_state):
        if gui:
            gui.update_board(current_state)
            pygame.display.update()

        current_node = mcts.run(current_state, search_agent)

        # Calculate how explored the board is
        explored_ratio = sum(current_state.board[0]) / (game_manager.size**2)
        # Use more exploration early in the game, less later
        dynamic_c = mcts.exploration_constant * (1 - explored_ratio)
        best_child = current_node.best_child(c_param=dynamic_c)

        if best_child is None:
            print("Warning: No best child found")
            break

        move = best_child.move
        
        # Assume state_tensor_canonical returns (canonical_board, extra_features, rotation)
        board_tensor, extra_features = current_node.state.state_tensor()

        # Retrieve the raw action distribution (a NumPy array).
        action_distribution = current_node.action_distribution(
            board_size=game_manager.size
        )

        # print("\nOriginal Action Distribution:")
        # visualize.plot_action_distribution(original_action_dist, game_manager.size)

        # Add the canonical state and action distribution to the replay buffer.
        if action_distribution is not None:
            rbuf.add_data_point(
                (
                    (board_tensor, extra_features),  # The canonical input state.
                    action_distribution,  # The canonicalized action distribution.
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
    placement_agents,
    play_game,
    sizes,
    epochs=1,
):
    move_count = []
    """Play a series of games, training and saving the model as specified."""
    gui = None
    # Initialize GUI
    if graphic_visualiser:
        pygame.init()
        pygame.display.set_caption("Battleship")
        gui = GUI(board_size)

    if placement_agents is None:
        placement_agent = PlacementAgent(
            board_size=board_size,
            ship_sizes=sizes,
            strategy="random",
        )

    if save_model:
        print("Saving model 0")
        search_agent.strategy.save_model(f"models/model_{0}.pth")

    for i in tqdm(range(number_actual_games)):
        if placement_agents is not None:
            placement_agent = placement_agents[i]
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
            # print(f"Finished game {i + 1} with {move_count[-1]} moves.")
            # print("Replay buffer length:", len(rbuf.data))
        visualize.print_rbuf(rbuf.get_batch(batch_size), 10, game_manager.size)
        if train_model:
            for _ in range(epochs):
                batch = rbuf.get_batch(batch_size)
                if len(batch) > 0 and len(rbuf.validation_set) > 0:
                    search_agent.strategy.train_model(batch)
                    search_agent.strategy.validate_model(rbuf.validation_set)

        # Save model at regular intervals
        if save_model and (i + 1) % (number_actual_games // M) == 0:
            print(f"Saving model at game {i + 1}")
            search_agent.strategy.save_model(f"models/model_{i + 1}.pth")

    if save_rbuf:
        rbuf.save_to_file(file_path="rbuf/rbuf.pkl")
    if train_model:
        search_agent.strategy.plot_metrics()

    if play_game:
        visualize.plot_fitness(move_count, game_manager.size)


def load_config(config_path="config/mcts_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Handle device configuration
    if config["device"] == "auto":
        if torch.cuda.is_available():
            config["device"] = "cuda"
        else:
            config["device"] = "cpu"

    return config


def run_mcts_inner_loop(
    game_manager,
    search_agent,
    simulations_number,  # Keep for backward compatibility
    exploration_constant,
    batch_size,
    device,
    sizes,
    placement_agents,
    epochs=1,
):
    """Train a search agent against a population of placing agents.
    Each search agent plays exactly one game against each placing agent.

    Args:
        game_manager: The game manager instance
        search_agent: The search agent to train
        simulations_number: Ignored (kept for backward compatibility)
        exploration_constant: MCTS exploration constant
        batch_size: Training batch size
        device: Computing device to use
        sizes: Ship sizes
        placement_agents: List of placing agents to train against
    """
    config = load_config()
    mcts = MCTS(
        game_manager,
        time_limit=config["mcts"]["time_limit"],
        exploration_constant=exploration_constant,
    )

    rbuf = RBUF(max_len=config["replay_buffer"]["max_size"])

    if config["replay_buffer"]["load_from_file"]:
        rbuf.load_from_file(file_path=config["replay_buffer"]["file_path"])

    train_models(
        game_manager,
        mcts,
        rbuf,
        search_agent,
        len(placement_agents),  # Number of games equals number of placing agents
        batch_size,
        M=config["training"]["save_interval"],
        device=device,
        graphic_visualiser=config["visualization"]["enable_graphics"],
        save_model=config["model"]["save"],
        train_model=config["model"]["train"],
        save_rbuf=config["replay_buffer"]["save_to_file"],
        board_size=game_manager.size,
        placement_agents=placement_agents,  # Use all placing agents
        play_game=True,  # Actually play the games
        sizes=sizes,
        epochs=epochs,
    )


def main():
    config = load_config()

    layer_config = json.load(open("ai/config.json"))

    net = ANET(
        board_size=config["board_size"],
        activation="relu",
        device=config["device"],
        layer_config=layer_config,
    )

    search_agent = SearchAgent(
        board_size=config["board_size"],
        strategy="nn_search",
        net=net,
        optimizer="adam",
        lr=config["training"]["learning_rate"],
    )

    game_manager = GameManager(size=config["board_size"])

    mcts = MCTS(
        game_manager,
        time_limit=config["mcts"]["time_limit"],
        exploration_constant=config["mcts"]["exploration_constant"],
    )

    rbuf = RBUF(max_len=config["replay_buffer"]["max_size"])

    if config["replay_buffer"]["load_from_file"]:
        rbuf.load_from_file(file_path=config["replay_buffer"]["file_path"])
        print("Loaded replay buffer from file. Length:", len(rbuf.data))

    train_models(
        game_manager,
        mcts,
        rbuf,
        search_agent,
        config["training"]["number_actual_games"],
        config["training"]["batch_size"],
        config["training"]["save_interval"],
        config["device"],
        config["visualization"]["enable_graphics"],
        config["model"]["save"],
        config["model"]["train"],
        config["replay_buffer"]["save_to_file"],
        config["board_size"],
        play_game=True,
        placement_agents=None,
        sizes=config["ship_sizes"],
        epochs=config["training"]["epochs"]
    )


if __name__ == "__main__":
    main()
