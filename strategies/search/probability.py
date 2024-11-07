from strategies.search.strategy import Strategy
from colorama import Fore, Style


class ProbabilisticStrategy(Strategy):

    def __init__(self, search_agent):
        super().__init__(search_agent)
        self.name = "probabilistic"

    def find_move(self):
        self.search_agent.move_count += 1
        probabilities = [0 for _ in range(self.search_agent.board_size ** 2)]
        possible_states = self.calculate_possible_states()

        # Loop through each unknown square and calculate probability
        for i in range(self.search_agent.board_size ** 2):
            if self.search_agent.board[0][i] == 0:  # Check only unknown squares
                probabilities[i] = self.calculate_probability(i, possible_states)

        # Find the square with the highest probability
        best_move = self.pick_highest_probability(probabilities)

        # For debugging purposes, print the probabilities
        self.print_probabilities(probabilities, best_move, possible_states)

        return best_move

    def calculate_possible_states(self):
        possible_states = 0

        for ship_size in self.search_agent.ship_sizes:
            # Iterate over all positions on the board
            for row in range(self.search_agent.board_size):
                for col in range(self.search_agent.board_size):
                    # Check if ship can be placed horizontally on non-miss tiles
                    if col + ship_size <= self.search_agent.board_size:
                        valid = True
                        for i in range(ship_size):
                            index = row * self.search_agent.board_size + (col + i)
                            if self.search_agent.board[2][index] == 2:  # Check if it's a miss
                                valid = False
                                break
                        if valid:
                            possible_states += 1

                    # Check if ship can be placed vertically on non-miss tiles
                    if row + ship_size <= self.search_agent.board_size:
                        valid = True
                        for i in range(ship_size):
                            index = (row + i) * self.search_agent.board_size + col
                            if self.search_agent.board[2][index] == 2:  # Check if it's a miss
                                valid = False
                                break
                        if valid:
                            possible_states += 1

        return possible_states

    def calculate_probability(self, index, possible_states):
        probability = 0
        row, col = divmod(index, self.search_agent.board_size)  # Convert index to row and column

        # For each ship, check all valid placements
        for ship_size in self.search_agent.ship_sizes:
            # Check horizontal placements
            for start_col in range(max(0, col - ship_size + 1),
                                   min(self.search_agent.board_size - ship_size + 1, col + 1)):
                start_index = row * self.search_agent.board_size + start_col
                if self.is_valid_placement(start_index, ship_size, 'horizontal'):
                    probability += 1

            # Check vertical placements
            for start_row in range(max(0, row - ship_size + 1),
                                   min(self.search_agent.board_size - ship_size + 1, row + 1)):
                start_index = start_row * self.search_agent.board_size + col
                if self.is_valid_placement(start_index, ship_size, 'vertical'):
                    probability += 1

        return probability / possible_states

    def is_valid_placement(self, index, ship_size, direction):
        row, col = divmod(index, self.search_agent.board_size)  # Convert index to row/col
        if direction == 'horizontal':
            if col + ship_size > self.search_agent.board_size:
                return False  # Ship would go off the board
            for i in range(ship_size):
                if self.search_agent.board[2][index + i] == 1 or self.search_agent.board[3][
                    index + i] == 1:  # Miss or sunk
                    return False
                if self.search_agent.board[1][index + i] == 1 and not self.in_same_ship(
                        index + i):  # Invalid hit placement
                    return False

        elif direction == 'vertical':
            if row + ship_size > self.search_agent.board_size:
                return False  # Ship would go off the board
            for i in range(ship_size):
                vertical_index = index + i * self.search_agent.board_size
                if self.search_agent.board[2][vertical_index] == 1 or self.search_agent.board[3][
                    vertical_index] == 1:  # Miss or sunk
                    return False
                if self.search_agent.board[1][vertical_index] == 1 and not self.in_same_ship(
                        vertical_index):  # Invalid hit placement
                    return False

        return True

    def in_same_ship(self, index):
        # Check if this hit is part of the same ship (basic assumption for now, needs more sophisticated handling)
        return True

    def pick_highest_probability(self, probabilities):
        max_prob = -1
        best_move = None
        for i in range(self.search_agent.board_size ** 2):
            if probabilities[i] > max_prob:
                max_prob = probabilities[i]
                best_move = i
        return best_move

    def print_probabilities(self, probabilities, best_move, possible_states):
        print(self.search_agent.board)
        # Print the probabilities in a grid format
        print("There are {} possible states.".format(possible_states))
        print("Probabilities: ")
        for row in range(self.search_agent.board_size):
            for col in range(self.search_agent.board_size):
                idx = row * self.search_agent.board_size + col
                prob = probabilities[idx]
                if idx == best_move:
                    # Highlight best move with a distinct color
                    print(f"{Fore.MAGENTA + Style.BRIGHT}{prob:.2f}{Style.RESET_ALL}", end=" ")
                else:
                    color = self.get_color(prob)
                    print(f"{color}{prob:.2f}{Style.RESET_ALL}", end=" ")
            print()  # Move to the next row after printing one row of the grid

    def get_color(self, prob):
        """
        Returns a color based on the probability value.
        More colors are added to represent finer probability ranges.
        """
        if prob >= 0.9:
            return Fore.RED + Style.BRIGHT  # Bright Red for highest probability
        elif prob >= 0.75:
            return Fore.RED
        elif prob >= 0.6:
            return Fore.YELLOW + Style.BRIGHT  # Bright Yellow for moderately high probability
        elif prob >= 0.5:
            return Fore.YELLOW
        elif prob >= 0.4:
            return Fore.GREEN + Style.BRIGHT  # Bright Green for moderate probability
        elif prob >= 0.25:
            return Fore.GREEN
        elif prob >= 0.1:
            return Fore.BLUE + Style.BRIGHT  # Bright Blue for low probability
        else:
            return Fore.BLUE  # Dark Blue for lowest probability
