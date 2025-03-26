import random
from game_logic.ship import Ship
from strategies.placing.strategy import Strategy


class CustomPlacing(Strategy):

    def __init__(self, placing_agent, chromosome=None):
        super().__init__(placing_agent)
        self.name = "random"
        self.chromosome = chromosome

    def place_ships(self):
        """
        Place ships according to the provided chromosome.
        If no chromosome is provided, place ships randomly.
        """
        if self.chromosome is None:
            self.place_ships_randomly()
        else:
            for i, (x, y, direction) in enumerate(self.chromosome):
                size = self.placing_agent.ship_sizes[i]
                ship = Ship(size, self.placing_agent.board_size, x, y, direction)
                if self.placing_agent.check_valid_placement(ship):
                    self.placing_agent.ships.append(ship)
                else:
                    print("Invalid placement")

    def place_ships_randomly(self):
        """
        Place ships randomly. For each ship, try random placements until a valid one is found.
        Also, store the generated placements in self.chromosome.
        """
        self.chromosome = []
        for i, size in enumerate(self.placing_agent.ship_sizes):
            placed = False
            while not placed:
                x, y, direction = self.random_ship_placement(size)
                ship = Ship(size, self.placing_agent.board_size, x, y, direction)
                if self.placing_agent.check_valid_placement(ship):
                    self.placing_agent.ships.append(ship)
                    self.chromosome.append((x, y, direction))
                    placed = True

    def random_ship_placement(self, size):
        """
        Generate a random placement for a ship of a given size.
        Returns a tuple (x, y, direction) where:
          - direction 0 means horizontal,
          - direction 1 means vertical.
        """
        # Randomly pick a direction (avoid using the name 'dir' to not shadow built-ins)
        direction_choice = random.randint(0, 1)
        board_size = self.placing_agent.board_size

        # Based on the direction, limit the starting point so that the ship fits on the board.
        if direction_choice == 0:  # Horizontal
            x = random.randint(0, board_size - 1)
            y = random.randint(0, board_size - size)
        else:  # Vertical
            x = random.randint(0, board_size - size)
            y = random.randint(0, board_size - 1)

        return x, y, direction_choice
