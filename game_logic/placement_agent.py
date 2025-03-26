import random
from colorama import Fore, Style
from game_logic.ship import Ship
from strategies.placing.custom import CustomPlacing
from strategies.placing.random import RandomPlacing
from strategies.placing.uniform_spread import UniformSpreadPlacing


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
        elif strategy == "chromosome":
            return CustomPlacing(self, chromosome)
        elif strategy == "uniform_spread":
            return UniformSpreadPlacing(self)
        else:
            raise ValueError("Unknown placing strategy")

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
        Adjust ship placements given a board state.

        The board is a list of four binary arrays of length board_size*board_size:
          board[0] - explored cells,
          board[1] - hits,
          board[2] - misses,
          board[3] - sunk.

        The new placements must obey two rules:
          1. If any cell has been hit, then some ship in the new configuration must cover that hit.
          2. For a ship that is not sunk, every cell in its candidate placement that is not a hit
             must be unexplored (i.e. the player has not fired there).

        Sunk ships remain fixed.
        """
        candidate_map = self._generate_candidate_map(board)
        chosen_config = self._sample_configuration(candidate_map, board)
        if chosen_config is None:
            # Handle failure as needed (e.g. do nothing or raise an error)
            return

        self._update_ships(chosen_config)

    def _generate_candidate_map(self, board):
        """
        Build and return a dictionary mapping each ship to its list of valid candidate placements.
        For sunk ships (any cell in ship.indexes is sunk) we only allow the current placement.
        For non-sunk ships we generate all placements (horizontal and vertical) that do not:
        - Overlap any cell that is marked as a miss (board[2]) or sunk (board[3]).
        - Place a ship on an explored cell unless that cell is a hit.
        Additionally, if a ship has one or more hit cells (i.e. it's been partially damaged
        but not sunk), then any candidate placement must cover those hit cells.
        """
        candidate_map = {}
        board_size = self.board_size

        # Get all hit cells that aren't part of sunk ships
        all_hits = {
            i
            for i, (hit, sunk) in enumerate(zip(board[1], board[3]))
            if hit == 1 and sunk == 0
        }

        for ship in self.ships:
            candidates = []
            # For sunk ships, only allow the current placement.
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

            # Get hits that are specifically on this ship
            ship_hits = {i for i in ship.indexes if board[1][i] == 1}

            for direction in [0, 1]:
                if direction == 0:
                    # Horizontal placements
                    for row in range(board_size):
                        for col in range(board_size - ship.size + 1):
                            temp_ship = Ship(ship.size, board_size, col, row, direction)
                            candidate_indexes = temp_ship.indexes
                            if not self._is_candidate_valid(candidate_indexes, board):
                                continue
                            # Must cover ship's own hits, but can also cover other hits
                            if ship_hits and not ship_hits.issubset(candidate_indexes):
                                continue
                            # Prioritize candidates that cover any hits
                            if set(candidate_indexes) & all_hits:
                                candidates.insert(
                                    0,
                                    {
                                        "col": col,
                                        "row": row,
                                        "direction": direction,
                                        "indexes": candidate_indexes,
                                    },
                                )
                            else:
                                candidates.append(
                                    {
                                        "col": col,
                                        "row": row,
                                        "direction": direction,
                                        "indexes": candidate_indexes,
                                    }
                                )
                else:
                    # Vertical placements
                    for col in range(board_size):
                        for row in range(board_size - ship.size + 1):
                            temp_ship = Ship(ship.size, board_size, col, row, direction)
                            candidate_indexes = temp_ship.indexes
                            if not self._is_candidate_valid(candidate_indexes, board):
                                continue
                            if ship_hits and not ship_hits.issubset(candidate_indexes):
                                continue
                            if set(candidate_indexes) & all_hits:
                                candidates.insert(
                                    0,
                                    {
                                        "col": col,
                                        "row": row,
                                        "direction": direction,
                                        "indexes": candidate_indexes,
                                    },
                                )
                            else:
                                candidates.append(
                                    {
                                        "col": col,
                                        "row": row,
                                        "direction": direction,
                                        "indexes": candidate_indexes,
                                    }
                                )

            # Shuffle the candidates to randomize placement preferences
            random.shuffle(candidates)
            candidate_map[ship] = candidates

        return candidate_map

    def _is_candidate_valid(self, candidate_indexes, board):
        """
        Determine whether a candidate placement is valid given the board state.

        For each cell index in candidate_indexes:
          - It must NOT be marked as a miss (board[2]==1) or as sunk (board[3]==1).
          - If the cell is NOT a hit (board[1]!=1), it must be unexplored (board[0]==0).

        This ensures that:
          • You never place a ship over a cell the player has confirmed as a miss.
          • For cells that aren't hits, you do not put a ship into an already-explored cell.
          • Hit cells (which are explored by definition) are allowed, so that the global constraint
            of "a ship must still be there" is met.
        """
        for idx in candidate_indexes:
            # Disallow placements on misses or sunk cells.
            if board[2][idx] == 1 or board[3][idx] == 1:
                return False
            # For non-hit cells, require that the cell is unexplored.
            if board[1][idx] != 1 and board[0][idx] == 1:
                return False
        return True

    def _sample_configuration(self, candidate_map, board):
        """
        Sample a complete configuration via randomized backtracking.

        In addition to avoiding candidate overlaps, we enforce the global constraint:
        every hit cell (i.e. where board[1]==1 and not sunk) must be covered by at least one ship.
        """
        ships_list = self.ships
        # Identify all hit cells that have not been sunk.
        global_hits = {
            i for i, hit in enumerate(board[1]) if hit == 1 and board[3][i] != 1
        }

        def backtrack(i, current_config, used_indexes):
            if i == len(ships_list):
                # Collect all indexes from the chosen candidate placements.
                all_indices = set()
                for cand in current_config:
                    all_indices.update(cand["indexes"])
                # Global constraint: every hit cell must be covered.
                if global_hits.issubset(all_indices):
                    return current_config.copy()
                else:
                    return None
            ship = ships_list[i]
            # Randomize candidate order.
            candidates = candidate_map[ship][:]
            random.shuffle(candidates)
            for cand in candidates:
                cand_set = set(cand["indexes"])
                if cand_set & used_indexes:
                    continue  # Overlap detected.
                current_config.append(cand)
                result = backtrack(i + 1, current_config, used_indexes | cand_set)
                if result is not None:
                    return result
                current_config.pop()
            return None

        return backtrack(0, [], set())

    def _update_ships(self, chosen_config):
        """
        Update each ship's attributes using the chosen configuration candidate.
        Also update the flattened list of ship indexes.
        """
        for ship, cand in zip(self.ships, chosen_config):
            ship.col = cand["col"]
            ship.row = cand["row"]
            ship.direction = cand["direction"]
            ship.indexes = cand["indexes"]

        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [i for sublist in self.list_of_ships for i in sublist]

        return self.list_of_ships


if __name__ == "__main__":
    # Example usage
    board_size = 5
    ship_sizes = [2, 3, 4]
    strategy = "uniform_spread"
    for i in range (5):
        agent = PlacementAgent(board_size, ship_sizes, strategy)
        agent.show_ships()
        print("\n")


