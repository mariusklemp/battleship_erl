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

    print("Ships to look for", current_state.placing.list_of_ships)
    move_count = 0

    while not game_manager.is_terminal(current_state):
        if gui:
            gui.update_board(current_state)
            pygame.display.update()
        visualize.show_board(current_state.board, board_size=game_manager.size)
        current_node = mcts.run(current_state, search_agent)
        best_child = current_node.best_child(c_param=0)

        move = best_child.move

        # Get both board tensor and extra features
        board_tensor, extra_features = current_node.state.state_tensor()

        # Add move to rbuf with both board and extra features
        # visualize.plot_action_distribution(
        #    current_node.action_distribution(board_size=game_manager.size),
        #    game_manager.size,
        # )
        rbuf.add_data_point(
            (
                (board_tensor, extra_features),  # Input tuple containing both tensors
                current_node.action_distribution(board_size=game_manager.size),
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
):
    move_count = []
    """Play a series of games, training and saving the model as specified."""
    gui = None
    # Initialize GUI
    if graphic_visualiser:
        pygame.init()
        pygame.display.set_caption("Battleship")
        gui = GUI(board_size)

    for i in tqdm(range(number_actual_games)):
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
            for _ in range(20):
                search_agent.strategy.train(batch, rbuf.get_validation_set())

        if save_model and ((i + 1) % (number_actual_games / M) == 0):
            search_agent.strategy.save_model(f"models/model_{i + 1}.pth")

    if save_rbuf:
        rbuf.save_to_file(file_path="rbuf/rbuf.pkl")
    if train_model:
        search_agent.strategy.plot_metrics()
    mcts.print_tree()
    visualize.plot_fitness(move_count, game_manager.size)


def main(
    board_size,
    sizes,
    strategy_placement,
    strategy_search,
    simulations_number,
    exploration_constant,
    M,
    number_actual_games,
    batch_size,
    device,
    load_rbuf,
    graphic_visualiser,
    save_model,
    train_model,
    save_rbuf,
):
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
    )


if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    main(
        board_size=3,
        sizes=[2, 1],
        strategy_placement="random",
        strategy_search="nn_search",
        simulations_number=1000,
        exploration_constant=1.41,
        M=10,
        number_actual_games=2,
        batch_size=100,
        device="cpu",
        load_rbuf=False,
        graphic_visualiser=False,
        save_model=False,
        train_model=False,
        save_rbuf=False,
    )
