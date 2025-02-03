import torch


class GameState:
    def __init__(self, board, move_count, placing):
        self.board = board
        self.move_count = move_count
        self.placing = placing

    def state_tensor(self):
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)
