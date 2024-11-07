import random

from game_logic.ship import Ship


class RandomPlacing():
    def __init__(self, placing_agent):
        self.placing_agent = placing_agent

    def place_ships(self):
        for size in self.placing_agent.ship_sizes:
            placed = False
            while not placed:
                # Create a new ship
                x, y, direction = self.random_ship_placement()

                ship = Ship(size, self.placing_agent.board_size, x, y, direction)

                # Check if the ship can be placed
                if self.placing_agent.check_valid_placement(ship):
                    self.placing_agent.ships.append(ship)
                    placed = True

    def random_ship_placement(self):
        # Fix this according to board size
        x, y = random.randint(0, self.placing_agent.board_size - 1), random.randint(0,
                                                                                    self.placing_agent.board_size - 1)
        direction = random.randint(0, 1)
        return x, y, direction
