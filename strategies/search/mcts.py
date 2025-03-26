from strategies.search.strategy import Strategy


class MCTSStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "mcts"
        self.mcts = None

    def find_move(self, state, topp=False):

        
        current_node = self.mcts.run(state, self.search_agent)
        # Picks the best move fully exploitation
        best_child = current_node.best_child(c_param=0)

        move = best_child.move
        return move

    def set_mcts(self, mcts):
        self.mcts = mcts
