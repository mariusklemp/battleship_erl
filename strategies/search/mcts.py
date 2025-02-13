import visualize
from strategies.search.strategy import Strategy
from visualize import plot_action_distribution, show_board


class MCTSStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "mcts"
        self.mcts = None

    def find_move(self, state):

        current_node = self.mcts.run(state, self.search_agent)
        # Picks the best move fully exploitation
        best_child = current_node.best_child(c_param=0)

        print("__Board__")
        state.placing.show_ships()
        # show_board(current_node.state.board, self.search_agent.board_size)
        #   action_distribution = current_node.action_distribution(
        #    board_size=self.search_agent.board_size
        # )
        # plot_action_distribution(action_distribution, self.search_agent.board_size)

        move = best_child.move
        return move

    def set_mcts(self, mcts):
        self.mcts = mcts
