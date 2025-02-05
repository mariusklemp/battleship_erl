from strategies.search.strategy import Strategy
from visualize import plot_action_distribution, show_board


class MCTSStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "mcts"
        self.mcts = None

    def find_move(self, state):
        self.search_agent.move_count += 1

        best_child = self.mcts.run(state, self.search_agent)

        # Visualize
        print("__Board__")
        state.placing.show_ships()
        show_board(best_child.state, self.search_agent.board_size)
        action_distribution = best_child.action_distribution(board_size=self.search_agent.board_size)
        plot_action_distribution(action_distribution, self.search_agent.board_size)

        move = best_child.move
        return move

    def set_mcts(self, mcts):
        self.mcts = mcts
