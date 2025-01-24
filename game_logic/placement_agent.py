import numpy as np
from colorama import Fore, Style, init

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
        self.ship_sizes = ship_sizes  # Sizes of the ships to place
        self.ships = []  # List to store placed ships
        # Board state as 2D array
        self.board = np.zeros(
            (self.board_size, self.board_size)
        )  # Initially empty board

        self.strategy = self.init_strategy(strategy, chromosome)

        self.strategy.place_ships()

        self.list_of_ships = [ship.indexes for ship in self.ships]
        self.indexes = [index for sublist in self.list_of_ships for index in sublist]

    def init_strategy(self, strategy, chromosome):
        if strategy == "random":
            return RandomPlacing(self)
        elif strategy == "nn_placing":
            return NNPlacing(self)
        elif strategy == "custom":
            return CustomPlacing(self, chromosome)

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
        # for row in range(self.board_size):
        #     print(" ".join(indexes[row * self.board_size:(row + 1) * self.board_size]))

        # print("\n")
