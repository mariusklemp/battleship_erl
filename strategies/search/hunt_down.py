import random

from strategies.search.strategy import Strategy


class HuntDownStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "hunt_down"

    def find_move(self):
        self.search_agent.move_count += 1

        sunk = {i for i, square in enumerate(self.search_agent.board[3]) if square == 1}
        hits_not_sunk = [i for i, square in enumerate(self.search_agent.board[1]) if square == 1 and i not in sunk]
        unknown = [i for i, square in enumerate(self.search_agent.board[0]) if square == 0]

        if hits_not_sunk:  # Hunt mode: Finish off the ship
            for hit in hits_not_sunk:

                row, col = divmod(hit, self.search_agent.board_size)
                potential_moves = [
                    (row - 1, col),  # Up
                    (row + 1, col),  # Down
                    (row, col - 1),  # Left
                    (row, col + 1)  # Right
                ]

                for r, c in potential_moves:
                    if 0 <= r < self.search_agent.board_size and 0 <= c < self.search_agent.board_size:
                        index = r * self.search_agent.board_size + c
                        if self.search_agent.board[0][index] == 0:  # Unknown cell
                            return index

        if unknown:  # Random mode: no hits found, make a random move
            return random.choice(unknown)
