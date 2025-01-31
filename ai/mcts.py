import random
import math
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, state, parent=None, move=None, untried_moves=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.fitness = 0
        self.visits = 0
        self.untried_moves = (
            untried_moves  # To be initialized based on the game manager
        )

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.fitness += result

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        best_score = -float("inf")
        best_moves = []
        for child_node in self.children:

            # Calculate score using UCB1 formula
            move_score = child_node.fitness / child_node.visits + c_param * math.sqrt(
                math.log(self.visits / child_node.visits)
            )

            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]
            elif move_score == best_score:
                best_moves.append(child_node)

        # Return one of the best moves randomly
        choice = random.choice(best_moves)

        if choice.move == None:
            print("Warning: No move found")

        return choice

    def action_distribution(self, board_size=5):
        size = board_size**2
        distribution = np.zeros(size)
        total_child_visits = sum([child.visits for child in self.children])
        for x in range(board_size):
            for child in self.children:
                if child.move == x:
                    # Normalize the visits to a 0-1 scale
                    distribution[x] = child.visits / total_child_visits
        return distribution

    def value(self):
        return self.fitness / self.visits


class MCTS:
    def __init__(self, game_manager, simulations_number=50, exploration_constant=1.41):
        self.game_manager = game_manager
        self.root_node = None
        self.current_node = None
        self.actor = None
        self.simulations_number = simulations_number
        self.exploration_constant = exploration_constant

    def select_node(self, node):
        while not node.is_fully_expanded():
            return self.expand_node(node)

        while not self.game_manager.is_terminal(node.state):
            if not node.is_fully_expanded():
                return self.expand_node(node)
            else:
                node = node.best_child(c_param=2)

        return node

    def expand_node(self, node):
        placing = self.generate_random_placing()
        move = random.choice(node.untried_moves)
        if move == None:
            print("Warning: No move found")
        node.untried_moves.remove(move)
        next_state = self.game_manager.next_state(node.state, move, placing)
        legal_moves = self.game_manager.get_legal_moves(next_state)
        child_node = Node(next_state, parent=node, move=move, untried_moves=legal_moves)
        node.add_child(child_node)
        return child_node

    def simulate(self, node):
        current_state = node.state
        placing = self.generate_random_placing()
        while not self.game_manager.is_terminal(current_state):
            best_move = self.actor.strategy.find_move(current_state)
            # best_move = random.choice(self.game_manager.get_legal_moves(current_state))
            current_state = self.game_manager.next_state(
                current_state, best_move, placing
            )
        return current_state.move_count

    def backpropagate(self, node, move_count):
        while node is not None:
            node.visits += 1
            node.fitness += (
                self.actor.board_size**2 - move_count
            ) / self.actor.board_size**2

            node = node.parent

    def run(
        self,
        current_state,
        actor,
    ):
        self.actor = actor
        # Check if we're starting a new game or continuing from an existing state
        if self.root_node is None:
            # Starting a new game
            self.root_node = Node(current_state)
            self.current_node = self.root_node

        elif self.equal(self.root_node.state, current_state):
            self.current_node = self.root_node
        else:
            # Continuing from an existing state: find the child node that matches the current state
            matching_child = next(
                (
                    child
                    for child in self.current_node.children
                    if self.equal(child.state, current_state)
                ),
                None,
            )

            if matching_child:
                # If a matching child is found, make it the new current node
                self.current_node = matching_child
            else:
                # Find the move that led to this state
                last_move = self.find_last_move(self.current_node, current_state)

                # Create a new node with the correct move
                new_node = Node(current_state, parent=self.current_node, move=last_move)

                self.current_node.add_child(new_node)
                self.current_node = new_node
        if self.current_node.untried_moves == None:
            self.current_node.untried_moves = self.game_manager.get_legal_moves(
                current_state
            )

        for _ in range(self.simulations_number):
            node = self.select_node(self.current_node)
            result = self.simulate(node)
            self.backpropagate(node, result)

        return self.current_node.best_child(c_param=self.exploration_constant)

    def equal(self, state1, state2):
        return state1.board == state2.board

    def print_tree(self, node=None, indent="", last=True):
        if node is None:
            node = self.root_node

        print(indent, end="")
        if last:
            print("└─", end="")
            indent += "  "
        else:
            print("├─", end="")
            indent += "| "

        print(f"Prev Move: {node.move}, fitness/Visits: {node.fitness}/{node.visits}")

        for i, child in enumerate(node.children):
            self.print_tree(child, indent, i == len(node.children) - 1)

    def generate_random_placing(self):
        # Initialize a blank board
        board_size = self.game_manager.size
        board = np.zeros((board_size, board_size), dtype=int)

        ship_sizes = self.game_manager.placing.ship_sizes  # Sizes of ships to place
        ships = []

        for ship_size in ship_sizes:
            placed = False
            while not placed:
                # Randomly choose orientation (horizontal or vertical)
                orientation = random.choice(["horizontal", "vertical"])

                if orientation == "horizontal":
                    row = random.randint(0, board_size - 1)
                    col = random.randint(0, board_size - ship_size)
                    indexes = [(row, col + i) for i in range(ship_size)]
                else:
                    row = random.randint(0, board_size - ship_size)
                    col = random.randint(0, board_size - 1)
                    indexes = [(row + i, col) for i in range(ship_size)]

                # Check if the placement is valid (no overlap)
                if all(board[x][y] == 0 for x, y in indexes):
                    # Mark the positions on the board
                    for x, y in indexes:
                        board[x][y] = 1

                    ships.append(indexes)
                    placed = True

        # Create a placement object with the indexes of the ships
        class ShadowPlacing:
            def __init__(self, indexes):
                self.indexes = [i for ship in indexes for i in ship]

        return ShadowPlacing(ships)

    def find_last_move(self, parent_node, current_state):
        """Finds the move that led from parent_node.state to current_state"""
        for move in self.game_manager.get_legal_moves(parent_node.state):
            if self.equal(
                self.game_manager.next_state(
                    parent_node.state, move, self.game_manager.placing
                ),
                current_state,
            ):
                return move
        return None  # Should not happen in normal execution
