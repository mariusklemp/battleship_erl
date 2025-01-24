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
        self.wins = 0
        self.visits = 0
        self.untried_moves = (
            untried_moves  # To be initialized based on the game manager
        )

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        best_score = -float("inf")
        best_moves = []
        current_player = 0
        for child_node in self.children:
            if self.state.current_player == "red":
                current_player = 1
            else:
                current_player = -1

            # Calculate score using UCB1 formula
            move_score = (
                current_player * child_node.wins / child_node.visits
                + c_param * math.sqrt(math.log(self.visits / child_node.visits))
            )

            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]
            elif move_score == best_score:
                best_moves.append(child_node)

        # Return one of the best moves randomly
        return random.choice(best_moves)

    def action_distribution(self, board_size=5):
        size = board_size**2
        distribution = np.zeros(size)
        total_child_visits = sum([child.visits for child in self.children])
        for x in range(board_size):
            for y in range(board_size):
                for child in self.children:
                    if child.move == (x, y):
                        # Normalize the visits to a 0-1 scale
                        distribution[x * board_size + y] = (
                            child.visits / total_child_visits
                        )
        return distribution

    def value(self):
        return self.wins / self.visits


class MCTS:
    def __init__(self, game_manager, actor):
        self.game_manager = game_manager
        self.root_node = None
        self.current_node = None
        self.actor = actor

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
        current_state = node.state
        while not self.game_manager.is_terminal(current_state):
            best_move = self.actor.default_policy(current_state)
            # best_move = random.choice(self.game_manager.get_legal_moves(current_state))
            current_state = self.game_manager.next_state(current_state, best_move)
        return self.game_manager.get_winner(current_state)

    def backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1

            if winner == "red":
                node.wins += 1
            elif winner == "blue":
                node.wins -= 1

            node = node.parent

    def run(self, current_state, simulations_number=1000):
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
                # If no matching child (e.g., a new move is made), create a new node and add it as a child
                new_node = Node(current_state)
                new_node.parent = self.current_node
                self.current_node.add_child(new_node)
                self.current_node = new_node
        if self.current_node.untried_moves == None:
            self.current_node.untried_moves = self.game_manager.get_legal_moves(
                current_state
            )

        for _ in range(simulations_number):
            node = self.select_node(self.current_node)
            result = self.simulate(node)
            self.backpropagate(node, result)

        return self.current_node.best_child()

    def equal(self, state1, state2):
        return (
            state1.board == state2.board
            and state1.current_player == state2.current_player
        )

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

        print(
            f"Prev Move: {node.move}, Wins/Visits: {node.wins}/{node.visits}, Player: {node.state.current_player}"
        )

        for i, child in enumerate(node.children):
            self.print_tree(child, indent, i == len(node.children) - 1)
