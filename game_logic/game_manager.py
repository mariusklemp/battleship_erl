from game_logic.game_state import GameState


class GameManager:
    """
    Class to manage the game state between a placing agent and a search agent
    Counts the number of moves and checks if the game is over
    """

    def __init__(self, size, placing):
        self.size = size
        self.placing = placing
        self.move_count = 0

    def initial_state(self):
        print("New game started")
        print("Placing ships")
        self.placing.new_placements()
        print("Setting board")
        board = [[0 for _ in range(self.size**2)] for _ in range(4)]
        print("Setting move count")
        print("Setting remaining ships", self.placing.ship_sizes)
        return GameState(
            board=board, move_count=0, placing=self.placing, remaining_ships=self.placing.ship_sizes
        )

    def get_legal_moves(self, state):
        legal_moves = []
        for i in range(self.size**2):
            if state.board[0][i] == 0:
                legal_moves.append(i)

        return legal_moves

    def next_state(self, state, move):
        new_board = [row[:] for row in state.board]
        new_move_count = state.move_count
        new_board[0][move] = 1
        remaining_ships = state.remaining_ships.copy()
        if move in self.placing.indexes:
            #print("Hit!")
            new_board[1][move] = 1
            #print("Checking if ship is sunk")
            sunk, ship_size = self.check_ship_sunk(move, new_board)
            #print("Ship sunk:", sunk)
            #print("Ship size:", ship_size)
            if sunk:
                if ship_size in remaining_ships:
                    remaining_ships.remove(ship_size)
                else:
                    print(
                        "Warning: sunk ship size",
                        ship_size,
                        "not found in remaining_ships",
                    )
        else:
            new_board[2][move] = 1
        new_move_count += 1
        return GameState(new_board, new_move_count, self.placing, remaining_ships)

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

        # Return if the ship is sunk and the ship size
        return sunk, len(hit_ship)

    def is_terminal(self, state):
        all_sunk = True
        for i in self.placing.indexes:
            if state.board[0][i] == 0:
                all_sunk = False
        return all_sunk
