from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from ai.mcts import MCTS
from game_logic.game_manager import GameManager
from tqdm import tqdm
from ai.models import ANET
from RBUF import RBUF
import json
import visualize


def simulate_game(game_manager, search_agent, mcts, rbuf):
    """Simulate a Battleship game and return the move count."""

    current_state = game_manager.initial_state()

    while not game_manager.is_terminal(current_state):
        game_manager.show_board(current_state)
        best_child = mcts.run(current_state, search_agent)
        move = best_child.move

        # Add move to rbuf
        rbuf.add_data_point(
            (
                best_child.state.state_tensor(),
                best_child.action_distribution(board_size=game_manager.size),
            )
        )

        # self.mcts.print_tree()

        current_state = game_manager.next_state(
            current_state, move
        )

    return current_state.move_count


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
):

    move_count = []
    """Play a series of games, training and saving the model as specified."""
    for i in tqdm(range(number_actual_games)):
        #game_manager.placing.new_placements()
        #game_manager.placing.show_ships()

        move_count.append(
            simulate_game(
                game_manager,
                search_agent,
                mcts,
                rbuf,
            )
        )
        if train_model:
            batch = rbuf.get_training_set(batch_size)
            for _ in range(20):
                search_agent.strategy.train(batch, rbuf.get_validation_set())

        if save_model and ((i + 1) % (number_actual_games / M) == 0):
            search_agent.strategy.save_model(f"/models/model_{i+1}.pth")

    if save_rbuf:
        rbuf.save_to_file(file_path="rbuf/rbuf.pkl")
    if train_model:
        search_agent.strategy.plot_metrics()

    visualize.plot_fitness(move_count, 5)


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
    )

    search_agent = SearchAgent(
        board_size=board_size,
        ship_sizes=sizes,
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

    game_manager = GameManager(size=board_size, placing=placement_agent)

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
    )


if __name__ == "__main__":
    main(
        board_size=5,
        sizes=[2, 2, 1],
        strategy_placement="random",
        strategy_search="nn_search",
        simulations_number=250,
        exploration_constant=1.41,
        M=10,
        number_actual_games=50,
        batch_size=50,
        device="cpu",
        load_rbuf=False,
        graphic_visualiser=False,
        save_model=False,
        train_model=False,
        save_rbuf=False,
    )
