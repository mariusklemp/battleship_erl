class Ship:
    # Mapping ship size to types for clarity
    SHIP_TYPE = {
        5: "Carrier",
        4: "Battleship",
        3: "Cruiser",
        2: "Destroyer"
    }

    # Mapping numbers to directions for clarity
    DIRECTION_MAP = {
        0: "horizontal",
        1: "vertical",
    }

    def __init__(self, size, board_size, col=None, row=None, direction=None):
        self.board_size = board_size
        self.col, self.row, = col, row
        self.size = size
        self.direction = direction
        self.indexes = self.compute_indexes()

    def compute_indexes(self):
        indexes = []
        start_index = self.row * self.board_size + self.col

        if self.direction == 0:  # horizontal
            for i in range(self.size):
                indexes.append(start_index + i)
        elif self.direction == 1:  # vertical
            for i in range(self.size):
                indexes.append(start_index + i * self.board_size)

        return indexes

    def print_direction(self):
        # For clarity when printing the direction
        print(f"Direction: {Ship.DIRECTION_MAP[self.direction]}")

    def print_ship_type(self):
        # For clarity when printing the ship type
        print(f"Ship Type: {Ship.SHIP_TYPE[self.size]} ({self.size})")