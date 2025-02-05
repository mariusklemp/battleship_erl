import random

import numpy as np
from colorama import Fore, Style, init

from game_logic.ship import Ship
from strategies.placing.NNPlacing import NNPlacing
from strategies.placing.custom import CustomPlacing
from strategies.placing.random import RandomPlacing


class PlacementAgent:
    def __init__(
        self,
        board_size,
        ship_sizes,
        strategy,
        chromosome=[(0, 0, 0), (2, 9, 0), (2, 2, 1), (0, 1, 1), (5, 2, 1)],
    ):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.ships = []

        self.strategy = self.init_strategy(strategy, chromosome)
        self.strategy.place_ships()

        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [index for sublist in self.list_of_ships for index in sublist]

    def init_strategy(self, strategy, chromosome=None):
        if strategy == "random":
            return RandomPlacing(self)
        elif strategy == "nn_placing":
            return NNPlacing(self)
        elif strategy == "custom":
            return CustomPlacing(self, chromosome)

    def new_placements(self):
        self.ships = []
        self.strategy.place_ships()
        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [index for sublist in self.list_of_ships for index in sublist]

    def adjust_ship_placements(self, board):
        """Randomly adjust ship placements while ensuring:
        - Sunken ships remain fixed.
        - Hit ships stay in the hit area (but can slide/rotate).
        - Other ships move freely.
        """
        sunken_tiles = board[3]  # Cells occupied by fully sunk ships

        for ship in self.ships:
            # If the ship is sunk, do NOT move it
            if any(sunken_tiles[i] == 1 for i in ship.indexes):
                continue  # Skip adjusting this ship

            # Remove current ship indexes from the board temporarily
            self.list_of_ships.remove(ship.indexes)

            # Generate all possible valid ship positions
            valid_positions = self.get_valid_adjustments(ship, board)

            if valid_positions:
                # Select a completely random valid placement
                new_col, new_row, new_direction = random.choice(valid_positions)

                # Update ship position
                ship.col = new_col
                ship.row = new_row
                ship.direction = new_direction
                ship.indexes = ship.compute_indexes()  # Recompute indexes

            # Re-add the adjusted ship
            self.list_of_ships.append(ship.indexes)

        # Update index tracking
        self.indexes = [index for sublist in self.list_of_ships for index in sublist]

    def get_valid_adjustments(self, ship, board):
        """Find alternative valid positions for a ship while ensuring:
        -  Hit ships retain at least one hit cell in their new position.
        -  Ships do not go out of bounds or overlap with misses.
        """

        board_size = self.board_size
        hit_tiles = board[1]
        miss_tiles = board[2]
        sunken_tiles = board[3]

        valid_positions = []
        directions = [0, 1]  # 0: horizontal, 1: vertical

        # Identify which ship parts are hit
        hit_positions = [i for i in ship.indexes if hit_tiles[i] == 1]

        for direction in directions:
            for row in range(board_size):
                for col in range(board_size):
                    # Create a temporary ship at this position
                    temp_ship = Ship(ship.size, board_size, col, row, direction)
                    temp_indexes = temp_ship.compute_indexes()

                    # Ensure all positions are within board limits
                    if any(i >= board_size**2 for i in temp_indexes):
                        continue

                    # Ensure the ship does not overlap with a fully sunken ship
                    if any(sunken_tiles[i] == 1 for i in temp_indexes):
                        continue

                    # Ensure no part of the ship is placed on a missed shot
                    if any(miss_tiles[i] == 1 for i in temp_indexes):
                        continue

                    # If ship was hit, ensure at least one hit position remains in the ship
                    if hit_positions and not any(
                        i in temp_indexes for i in hit_positions
                    ):
                        continue

                    # Ensure the placement does not overlap with other ships
                    if any(i in self.indexes for i in temp_indexes):
                        continue

                    # If valid, add it to the list of possible positions
                    valid_positions.append((col, row, direction))

        return valid_positions

    def check_valid_placement(self, ship):
        possible = True
        for i in ship.indexes:
            # indexes must be within the board
            if i < 0 or i >= (self.board_size**2) - 1:
                possible = False
                break

            # Ships cannot behave like snake game
            new_row = i // self.board_size
            new_col = i % self.board_size
            if new_col != ship.col and new_row != ship.row:
                possible = False
                break

            # Check if the ship overlaps with another ship
            for other_ship in self.ships:
                if i in other_ship.indexes:
                    possible = False
                    break

        return possible

    # Define colors for each ship size
    SIZE_COLORS = {
        5: Fore.RED,
        4: Fore.GREEN,
        3: Fore.BLUE,
        2: Fore.YELLOW,
        1: Fore.CYAN,
    }

    def show_ships(self):
        # print("Current Ship Placements:")
        # Initialize the board with empty cells
        indexes = ["-" for _ in range(self.board_size**2)]

        # Iterate over each ship in list_of_ships
        for ship in self.list_of_ships:
            ship_size = len(ship)  # Determine the size of the ship
            color = self.SIZE_COLORS.get(
                ship_size, Fore.WHITE
            )  # Get color based on ship size

            # Place colored markers in the ship's positions
            for position in ship:
                indexes[position] = f"{color}X{Style.RESET_ALL}"

        # Print each row with colored ship indicators
        for row in range(self.board_size):
            print(
                " ".join(indexes[row * self.board_size : (row + 1) * self.board_size])
            )
