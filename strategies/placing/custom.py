from game_logic.ship import Ship


class CustomPlacing:
    def __init__(self, placing_agent, chromosome):
        """
        Initialize the CustomPlacing strategy with a chromosome.
        The chromosome is a list of tuples (x, y, direction) for each ship.
        """
        self.placing_agent = placing_agent
        self.chromosome = chromosome  # Store the chromosome in the agent

    def place_ships(self):
        """Place ships according to the chromosome."""
        for i, (x, y, direction) in enumerate(self.chromosome):
            size = self.placing_agent.ship_sizes[i]
            ship = Ship(size, self.placing_agent.board_size, x, y, direction)

            if self.placing_agent.check_valid_placement(ship):
                self.placing_agent.ships.append(ship)

