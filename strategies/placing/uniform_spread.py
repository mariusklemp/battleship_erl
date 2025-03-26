import numpy as np
from game_logic.ship import Ship
from strategies.placing.strategy import Strategy


class UniformSpreadPlacing(Strategy):
    def __init__(self, placing_agent):
        super().__init__(placing_agent)
        self.name = "uniform_spread"

    def place_ships(self):
        ship_sizes = self.placing_agent.ship_sizes.copy()
        np.random.shuffle(ship_sizes)
        board_size = self.placing_agent.board_size
        board = np.zeros((board_size, board_size))

        for size in ship_sizes:
            candidates = []
            best_distance = -1

            for x in range(board_size):
                for y in range(board_size):
                    for direction in [0, 1]:
                        ship = Ship(size, board_size, x, y, direction)
                        if self.placing_agent.check_valid_placement(ship):
                            distance = self.average_distance(ship)

                            if distance > best_distance:
                                best_distance = distance
                                candidates = [(x, y, direction)]
                            elif distance == best_distance:
                                candidates.append((x, y, direction))

            if candidates:
                np.random.shuffle(candidates)  # Add randomness
                chosen_position = candidates[0]
                ship = Ship(size, board_size, *chosen_position)
                self.placing_agent.ships.append(ship)
                self.mark_ship_on_board(ship, board)

    def average_distance(self, ship):
        new_center = np.mean([(idx % ship.board_size, idx // ship.board_size) for idx in ship.indexes], axis=0)

        existing_centers = []

        for existing_ship in self.placing_agent.ships:
            coords = [(idx % ship.board_size, idx // ship.board_size) for idx in existing_ship.indexes]
            center = np.mean(coords, axis=0)
            existing_centers.append(center)

        if not existing_centers:
            return np.inf

        dists = [np.linalg.norm(new_center - center) for center in existing_centers]
        return np.mean(dists)


    def mark_ship_on_board(self, ship, board):
        for x, y in [(idx % ship.board_size, idx // ship.board_size) for idx in ship.indexes]:
            board[y, x] = 1
