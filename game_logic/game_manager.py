from game_logic.game_state import GameState
import copy


class GameManager:
    """
    Class to manage the game state between a placing agent and a search agent
    Counts the number of moves and checks if the game is over
    """

    def __init__(self, size):
        self.size = size

    def initial_state(self, placing):
        board = [[0 for _ in range(self.size**2)] for _ in range(4)]

        return GameState(
            board=board,
            move_count=0,
            placing=placing,
            remaining_ships=placing.ship_sizes.copy(),
        )

    def get_legal_moves(self, state):
        legal_moves = []
        for i in range(self.size**2):
            if state.board[0][i] == 0:
                legal_moves.append(i)

        return legal_moves

    def next_state(self, state, move, sim_id=None):
        new_board = [row[:] for row in state.board]
        new_move_count = state.move_count
        new_board[0][move] = 1
        new_remaining_ships = state.remaining_ships.copy()

        if move in state.placing.indexes:
            new_board[1][move] = 1
            sunk, ship_size, hit_ship = self.check_ship_sunk(
                move, new_board, state.placing
            )

            if sunk:
                if hit_ship is None:
                    print("Warning: hit_ship is None")
                if ship_size in new_remaining_ships:
                    for i in hit_ship:
                        new_board[3][i] = 1
                    new_remaining_ships.remove(ship_size)
                else:
                    print(
                        "Warning: sunk ship size",
                        ship_size,
                        "not found in remaining_ships",
                    )
        else:
            new_board[2][move] = 1
        new_move_count += 1
        return GameState(new_board, new_move_count, state.placing, new_remaining_ships)

    def check_ship_sunk(self, move, board, placing):
        hit_ship = None
        sunk = True
        # Find the ship that was hit
        for ship in placing.ships:
            if move in ship.indexes:
                hit_ship = ship.indexes

        sunk = all(board[1][i] == 1 for i in hit_ship)

        # Return if the ship is sunk and the ship size
        return sunk, len(hit_ship), hit_ship

    def is_terminal(self, state):
        return len(state.remaining_ships) == 0
