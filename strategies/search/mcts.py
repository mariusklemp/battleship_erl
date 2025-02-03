from strategies.search.strategy import Strategy


class MCTSStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "mcts"
        self.mcts = None

    def find_move(self, state):
        print("MCTS move")
        state.placing.show_ships()

        print("Move count:", state.move_count)
        print("Board:", state.board)

        print("Search move count:", self.search_agent.move_count)

        self.search_agent.move_count += 1

        best_child = self.mcts.run(state, self.search_agent)
        #self.mcts.print_tree()
        move = best_child.move
        return move

    def set_mcts(self, mcts):
        self.mcts = mcts
