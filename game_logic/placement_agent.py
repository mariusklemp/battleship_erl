import random
from colorama import Fore, Style
from game_logic.ship import Ship
from strategies.placing.NNPlacing import NNPlacing
from strategies.placing.custom import CustomPlacing
from strategies.placing.random import RandomPlacing


class PlacementAgent:
    # Define colors for each ship size (used by show_ships)
    SIZE_COLORS = {
        5: Fore.RED,
        4: Fore.GREEN,
        3: Fore.BLUE,
        2: Fore.YELLOW,
        1: Fore.CYAN,
    }

    def __init__(self, board_size, ship_sizes, strategy, chromosome=None):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.ships = []  # List of Ship objects

        # Strategy is expected to populate self.ships.
        self.strategy = self.init_strategy(strategy, chromosome)
        self.strategy.place_ships()

        # Maintain a list-of-ships as lists of indexes and a flattened list of all indexes.
        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [i for sublist in self.list_of_ships for i in sublist]

    def init_strategy(self, strategy, chromosome=None):
        if strategy == "random":
            return RandomPlacing(self)
        elif strategy == "nn_placing":
            return NNPlacing(self)
        elif strategy == "custom":
            return CustomPlacing(self, chromosome)
        else:
            raise ValueError("Unknown strategy")

    def new_placements(self):
        self.ships = []
        self.strategy.place_ships()
        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [i for sublist in self.list_of_ships for i in sublist]

    def check_valid_placement(self, ship):
        possible = True
        for i in ship.indexes:
            if i < 0 or i >= (self.board_size**2):
                possible = False
                break

            new_row = i // self.board_size
            new_col = i % self.board_size
            if new_col != ship.col and new_row != ship.row:
                possible = False
                break

            for other_ship in self.ships:
                if other_ship is not ship and i in other_ship.indexes:
                    possible = False
                    break
        return possible

    def show_ships(self):
        """Display the current board with ships (cells with a ship show an 'X' colored by ship size)."""
        board = ["-" for _ in range(self.board_size**2)]
        for ship in self.list_of_ships:
            ship_size = len(ship)
            color = self.SIZE_COLORS.get(ship_size, Fore.WHITE)
            for pos in ship:
                board[pos] = f"{color}X{Style.RESET_ALL}"
        for row in range(self.board_size):
            print(" ".join(board[row * self.board_size : (row + 1) * self.board_size]))

    def adjust_ship_placements(self, board):
        """
        Given a board state (a list of four binary arrays of length board_size*board_size):
          board[0] - explored cells,
          board[1] - hits,
          board[2] - misses,
          board[3] - sunk.

        For each ship:
          - Sunken ships remain fixed.
          - Hit ships are placed so that every hit cell remains within the ship.
          - Free ships are only placed on unexplored cells.

        Instead of generating all configurations, we sample a single valid configuration
        (if one exists) and update the ship placements accordingly.
        """
        candidate_map = self._generate_candidate_map(board)
        chosen_config = self._sample_configuration(candidate_map)
        if chosen_config is None:
            # Handle the failure (e.g. do nothing or raise an error)
            return

        self._update_ships(chosen_config)

    def _generate_candidate_map(self, board):
        """
        Build and return a dictionary mapping each ship to its list of valid candidate placements.
        Each candidate is a dict with keys: 'col', 'row', 'direction', and 'indexes'.
        """
        candidate_map = {}
        board_size = self.board_size

        for ship in self.ships:
            candidates = []
            # If any cell in the ship is sunk, only the current placement is valid.
            if any(board[3][i] == 1 for i in ship.indexes):
                candidates.append(
                    {
                        "col": ship.col,
                        "row": ship.row,
                        "direction": ship.direction,
                        "indexes": ship.indexes,
                    }
                )
                candidate_map[ship] = candidates
                continue

            hit_positions = [i for i in ship.indexes if board[1][i] == 1]
            is_hit = len(hit_positions) > 0

            for direction in [0, 1]:
                if direction == 0:
                    # Horizontal placements: valid columns are 0 .. board_size - ship.size
                    for row in range(board_size):
                        for col in range(board_size - ship.size + 1):
                            temp_ship = Ship(ship.size, board_size, col, row, direction)
                            candidate_indexes = temp_ship.indexes

                            if not self._is_candidate_valid(
                                candidate_indexes, board, is_hit, hit_positions
                            ):
                                continue

                            candidates.append(
                                {
                                    "col": col,
                                    "row": row,
                                    "direction": direction,
                                    "indexes": candidate_indexes,
                                }
                            )
                else:  # Vertical placements
                    for col in range(board_size):
                        for row in range(board_size - ship.size + 1):
                            temp_ship = Ship(ship.size, board_size, col, row, direction)
                            candidate_indexes = temp_ship.indexes

                            if not self._is_candidate_valid(
                                candidate_indexes, board, is_hit, hit_positions
                            ):
                                continue

                            candidates.append(
                                {
                                    "col": col,
                                    "row": row,
                                    "direction": direction,
                                    "indexes": candidate_indexes,
                                }
                            )
            candidate_map[ship] = candidates

        return candidate_map

    def _is_candidate_valid(self, candidate_indexes, board, is_hit, hit_positions):
        """
        Check whether all cells in candidate_indexes satisfy:
          - No overlap with misses (board[2]) or sunken cells (board[3])
          - For a free ship (not hit): candidate cells must be unexplored (board[0] is 0)
          - For a hit ship: all previously hit cells must be included
        """
        for idx in candidate_indexes:
            if board[2][idx] == 1 or board[3][idx] == 1:
                return False
            if not is_hit and board[0][idx] == 1:
                return False

        if is_hit and not all(hp in candidate_indexes for hp in hit_positions):
            return False

        return True

    def _sample_configuration(self, candidate_map):
        """
        Instead of generating all configurations, sample a single valid configuration.
        This function performs a randomized backtracking search and returns the first
        complete configuration found (a list of candidate placements, one per ship).
        """
        ships_list = self.ships

        def backtrack(i, current_config, used_indexes):
            if i == len(ships_list):
                return current_config.copy()
            ship = ships_list[i]
            # Randomize the candidate order to sample a random configuration.
            candidates = candidate_map[ship][:]
            random.shuffle(candidates)
            for cand in candidates:
                cand_set = set(cand["indexes"])
                if cand_set & used_indexes:
                    continue  # Overlap detected
                current_config.append(cand)
                result = backtrack(i + 1, current_config, used_indexes | cand_set)
                if result is not None:
                    return result
                current_config.pop()
            return None

        return backtrack(0, [], set())

    def _update_ships(self, chosen_config):
        """
        Update each ship's attributes (col, row, direction, indexes) using the chosen configuration.
        Also update the flattened list of ship indexes.
        """
        for ship, cand in zip(self.ships, chosen_config):
            ship.col = cand["col"]
            ship.row = cand["row"]
            ship.direction = cand["direction"]
            ship.indexes = cand["indexes"]

        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [i for sublist in self.list_of_ships for i in sublist]
