import copy
import random
import math
import numpy as np


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
        """
        Compute the action distribution from MCTS node visits.

        :param board_size: The size of the Battleship board (assumed square).
        :return: A NumPy array representing the action probability distribution.
        """
        size = board_size ** 2
        distribution = np.zeros(size)  # Initialize the distribution

        # Compute total visits among all children.
        total_child_visits = sum(child.visits for child in self.children)

        if total_child_visits == 0:
            return distribution

        # Process each child and fill the distribution.
        for child in self.children:
            move_index = child.move  # Assuming move is an integer index in range(size)
            if 0 <= move_index < size:
                distribution[move_index] = child.visits / total_child_visits

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
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        next_state = self.game_manager.next_state(node.state, move)
        legal_moves = self.game_manager.get_legal_moves(next_state)
        child_node = Node(next_state, parent=node, move=move, untried_moves=legal_moves)
        node.add_child(child_node)
        return child_node

    def simulate(self, node):
        # Make a copy of the board to avoid modifying the original node
        current_state = copy.deepcopy(node.state)

        current_state.placing.adjust_ship_placements(current_state.board)

        while not self.game_manager.is_terminal(current_state):
            # Select a move randomly for now (or use a strategy)
            best_move = random.choice(self.game_manager.get_legal_moves(current_state))
            current_state = self.game_manager.next_state(current_state, best_move)

        return current_state.move_count

    def backpropagate(self, node, move_count):
        while node is not None:
            node.visits += 1
            node.fitness += (
                                    self.actor.board_size ** 2 - move_count
                            ) / self.actor.board_size ** 2

            node = node.parent

    def run(
            self,
            current_state,
            actor,
    ):

        self.actor = actor

        self.current_node = self.find_current_node(current_state)

        # Ensure that the current node has its legal moves
        if self.current_node.untried_moves is None:
            self.current_node.untried_moves = self.game_manager.get_legal_moves(
                current_state
            )

        # Simulate a number of games
        for _ in range(self.simulations_number):
            node = self.select_node(self.current_node)
            result = self.simulate(node)
            self.backpropagate(node, result)

        best_child = self.current_node.best_child(c_param=self.exploration_constant)

        # Now prune the tree so that the branch from the best action becomes the new tree.
        self.prune_tree()

        return best_child

    def find_current_node(self, current_state):
        # Check if we're starting a new game
        if self.root_node is None:
            self.root_node = Node(current_state)
            current_node = self.root_node

        # Check if the current state equals the root node's state
        elif self.equal(self.root_node.state, current_state):
            current_node = self.root_node

        else:
            matching_child = next(
                (
                    child
                    for child in self.current_node.children
                    if self.equal(child.state, current_state)
                ),
                None,
            )
            if matching_child:
                current_node = matching_child
            else:
                last_move = self.find_last_move(self.current_node, current_state)

                # Create a new node based on the last move
                new_node = Node(current_state, parent=self.current_node, move=last_move)

                self.current_node.add_child(new_node)
                current_node = new_node

        return current_node

    def find_last_move(self, parent_node, current_state):
        """Finds the move that led from parent_node.state to current_state"""
        for move in self.game_manager.get_legal_moves(parent_node.state):
            if self.equal(
                    self.game_manager.next_state(
                        parent_node.state, move
                    ),
                    current_state,
            ):
                return move
        return None  # Should not happen in normal execution

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

    def prune_tree(self):
        """
        Prunes the tree to only keep the subtree that branches out from the best action.
        This method re-roots the tree by selecting the best child (using c_param=0 for pure exploitation),
        detaching it from its parent, and setting it as the new root.
        """
        if self.root_node is None:
            return

        if len(self.root_node.children) == 0:
            return

        # Use c_param=0 to pick the child with the highest win rate (pure exploitation)
        best_child = self.root_node.best_child(c_param=0)
        if best_child is None:
            return

        # Detach best_child from its parent.
        best_child.parent = None
        # Set the best child as the new root.
        self.root_node = best_child
