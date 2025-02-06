from strategies.search.strategy import Strategy
from visualize import plot_action_distribution, show_board


class MCTSStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "mcts"
        self.mcts = None

    def find_move(self, state):
        self.search_agent.move_count += 1

        current_node = self.mcts.run(state, self.search_agent)
        best_child = current_node.best_child(c_param=self.mcts.exploration_constant)

        # Visualize
        print("__Board__")
        state.placing.show_ships()
        show_board(current_node.state, self.search_agent.board_size)
        action_distribution = current_node.action_distribution(
            board_size=self.search_agent.board_size
        )
        plot_action_distribution(action_distribution, self.search_agent.board_size)

        move = best_child.move
        return move

    def set_mcts(self, mcts):
        self.mcts = mcts
