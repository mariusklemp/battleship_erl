import random

from strategies.search.strategy import Strategy


class RandomStrategy(Strategy):
    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "random"

    def find_move(self, state):
        unknown = [i for i, square in enumerate(state.board[0]) if square == 0]
        if len(unknown) > 0:
            move = random.choice(unknown)
            return move
