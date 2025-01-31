import torch


class GameState:
    def __init__(self, board, move_count):
        self.board = board
        self.move_count = move_count

    def state_tensor(self):
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)


class GameManager:
    """
    Class to manage the game state between a placing agent and a search agent
    Counts the number of moves and checks if the game is over
    """

    def __init__(self, size):
        self.size = size
        self.board = [[0 for _ in range(self.size**2)] for _ in range(4)]
        self.game_over = False
        self.move_count = 0
        self.placing = None

    def initial_state(self, placing):
        # Update the board with the ships
        self.placing = placing
        return GameState(self.board, self.move_count)

    def get_legal_moves(self, state):
        legal_moves = []
        for i in range(self.size**2):
            if state.board[0][i] == 0:
                legal_moves.append(i)

        return legal_moves

    def next_state(self, state, move, placing):
        new_board = [row[:] for row in state.board]
        new_move_count = state.move_count
        new_board[0][move] = 1
        if move in placing.indexes:
            new_board[1][move] = 1
            self.check_ship_sunk(move, new_board)
        else:
            new_board[2][move] = 1
        new_move_count += 1
        return GameState(new_board, new_move_count)

    def play_turn(self):
        move = self.searching.strategy.find_move()

        # Update state based on the selected action
        try:
            # Check for illegal action
            assert self.is_move_possible(move), "Move has already been made"

            # Do the move
            self.make_move(move)

        except AssertionError as e:

            print("Illegal move", move)
            return False

        self.move_count += 1
        # Check if all ships are sunk
        self.check_game_over()

    def is_move_possible(self, move):

        if move is None:
            return False
        if self.board[0][move] != 0:
            return False
        return True

    def make_move(self, move):

        self.board[0][move] = 1  # Not unknown
        if move in self.placing.indexes:
            self.board[1][move] = 1  # Hit
            self.check_ship_sunk(move)
        else:
            self.board[2][move] = 1  # Miss

    def check_ship_sunk(self, move, board):
        hit_ship = None
        sunk = True

        # Find the ship that was hit
        for ship in self.placing.ships:
            if move in ship.indexes:
                hit_ship = ship.indexes

        # Check if the ship is sunk
        for i in hit_ship:
            if board[0][i] == 0:
                sunk = False
                break

        # If the ship is sunk, update the search board
        if sunk:
            for i in hit_ship:
                board[3][i] = 1

    def check_game_over(self):
        all_sunk = True
        for i in self.placing.indexes:
            if self.board[0][i] == 0:
                all_sunk = False
        self.game_over = all_sunk

    def is_terminal(self, state):
        all_sunk = True
        for i in self.placing.indexes:
            if state.board[0][i] == 0:
                all_sunk = False
        return all_sunk

    # Show - for unknown, X for hit, O for miss, S for sunk
    def show_board(self, state):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                index = i * self.size + j
                if state.board[0][index] == 0:
                    row += "- "
                elif state.board[3][index] == 1:
                    row += "S "
                elif state.board[1][index] == 1:
                    row += "X "
                elif state.board[2][index] == 1:
                    row += "O "

            print(row)
        print("\n")
