import copy
import random
import math
import numpy as np
import visualize

MIN_VISITS_THRESHOLD = 10


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

    def child_exists(self, move, state):
        for child in self.children:
            if child.move == move and child.state.board == state.board:
                return child
        return None

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
        if len(best_moves) < 1:
            print("Warning: No best moves found")
            return None
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
        size = board_size**2
        distribution = np.zeros(size)  # Initialize the distribution

        # Compute total visits among all children.
        total_child_visits = sum(child.visits for child in self.children)

        if total_child_visits < MIN_VISITS_THRESHOLD:  # e.g., 10
            return None  # Skip adding to RBUF if not enough visits

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
                node = node.best_child(c_param=self.exploration_constant)

        return node

    def expand_node(self, node):
        # Get the raw legal moves from the game manager.
        legal_moves = node.untried_moves
        # Use the pruning function to filter them based on ship-placement constraints.
        state = copy.deepcopy(node.state)
        # moves_to_expand = self.prune_moves(legal_moves, state)
        moves_to_expand = legal_moves
        # If no moves remain after pruning, fallback to the original list (or handle appropriately)
        if not moves_to_expand:
            moves_to_expand = legal_moves

        move = random.choice(moves_to_expand)
        node.untried_moves.remove(move)
        next_state = self.game_manager.next_state(state, move)

        next_legal_moves = self.game_manager.get_legal_moves(next_state)

        if move is None:
            print("Warning: Move is None", flush=True)

        child_node = node.child_exists(move, next_state)
        if child_node is None:
            child_node = Node(
                next_state, parent=node, move=move, untried_moves=next_legal_moves
            )
            node.add_child(child_node)
        return child_node

    def simulate(self, node, sim_id):
        current_state = node.state
        while not self.game_manager.is_terminal(current_state):
            legal_moves = self.game_manager.get_legal_moves(current_state)
            if len(legal_moves) == 0:
                print("No legal moves found", flush=True)
                break
            best_move = random.choice(legal_moves)
            current_state = self.game_manager.next_state(
                current_state, best_move, sim_id
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

        self.current_node = self.find_current_node(current_state)

        # Ensure that the current node has its legal moves
        if (
            self.current_node.untried_moves is None
            or len(self.current_node.untried_moves) == 0
        ):
            self.current_node.untried_moves = self.game_manager.get_legal_moves(
                current_state
            )

        # Simulate a number of games
        for i in range(self.simulations_number):
            current_state = copy.deepcopy(self.current_node.state)
            new_placing = copy.deepcopy(current_state.placing)

            # After adjusting placements, get the new ship sizes
            new_placing.adjust_ship_placements(current_state.board)

            current_state.placing = new_placing
            self.current_node.state = current_state

            node = self.select_node(self.current_node)

            # If board 1 contains a 1 in a cell that is not in placing.indexes, print
            if any(
                node.state.board[1][i] == 1 and i not in node.state.placing.indexes
                for i in range(len(node.state.board[1]))
            ):
                print(
                    "Warning: Board 1 contains a 1 in a cell that is not in placing.indexes",
                    flush=True,
                )

            result = self.simulate(node, sim_id=i)
            self.backpropagate(node, result)

        return self.current_node

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
                if last_move is None:
                    print("Warning: Last move is None", flush=True)

                # Create a new node based on the last move
                new_node = Node(current_state, parent=self.current_node, move=last_move)
                self.current_node.add_child(new_node)
                current_node = new_node

        return current_node

    def find_last_move(self, parent_node, current_state):
        """Finds the move that led from parent_node.state to current_state"""
        legal_moves = self.game_manager.get_legal_moves(parent_node.state)
        # Find which one of the legal moves that exist in current_state.board[0] but not in parent_node.state.board[0]
        for move in legal_moves:
            if (
                current_state.board[0][move] == 1
                and parent_node.state.board[0][move] == 0
            ):
                return move
        return None

    def equal(self, state1, state2):
        return (
            state1.board == state2.board
            and state1.remaining_ships == state2.remaining_ships
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

        fitness = round(node.fitness, 2)
        parent_visits = node.parent.visits if node.parent else node.visits

        print(
            f"Prev Move: {node.move}, fitness: {fitness}, Visits: {node.visits}/{parent_visits} Board: {node.state.board}"
        )

        for i, child in enumerate(node.children):
            self.print_tree(child, indent, i == len(node.children) - 1)

    def prune_moves(self, legal_moves, state):
        """
        Given a list of candidate moves (each as a tuple (row, col)) and the current state,
        return only those moves that are consistent with any remaining ship placement.
        """
        pruned = [
            move for move in legal_moves if self.is_possible_ship_location(state, move)
        ]

        return pruned

    def is_possible_ship_location(self, state, move):
        """
        Check if a candidate move (cell) could be part of any remaining ship.

        Parameters:
            state: an object that must have:
                - state.board: a list of 4 lists (layers) of length board_size**2.
                Layer 0: Explored cells (1 for explored, 0 for unexplored)
                Layer 1: Hit cells (1 for hit, 0 otherwise)
                Layer 2: Misses (1 for miss, 0 otherwise)
                Layer 3: Sunken ships (1 for sunken ship cell, 0 otherwise)
                - state.remaining_ships: a list of ship sizes that have not yet been sunk.
            move: an integer (0 to board_size**2 - 1) representing the candidate cell in row‑major order.

        Returns:
            True if there exists at least one legal placement (horizontal or vertical) for any of the
            remaining ships that includes the candidate cell, given that each cell in the placement is either
            unexplored (layer 0 is 0) or already a confirmed hit (layer 0 is 1 and layer 1 is 1). Otherwise, returns False.
        """
        board = state.board
        remaining_ships = state.remaining_ships  # e.g., [2, 2, 1]
        total_cells = len(board[0])
        board_size = int(math.sqrt(total_cells))

        # Convert move (an integer) to (row, col)
        r = move // board_size
        c = move % board_size

        # Candidate cell is ruled out if:
        # 1. It is explored but not a hit.
        # 2. It is marked as a miss or as part of a sunken ship.
        if board[0][move] == 1 and board[1][move] == 0:
            return False
        if board[2][move] == 1 or board[3][move] == 1:
            return False

        # For each remaining ship size, check for at least one valid placement that includes this cell.
        for size in remaining_ships:
            # --- Horizontal placements ---
            start_c_min = max(0, c - size + 1)
            start_c_max = min(c, board_size - size)
            for start_c in range(start_c_min, start_c_max + 1):
                valid = True
                # Check the horizontal segment in row r from start_c to start_c+size-1.
                for j in range(start_c, start_c + size):
                    idx = r * board_size + j
                    if board[2][idx] == 1 or board[3][idx] == 1:
                        valid = False
                        break
                    if board[0][idx] == 1 and board[1][idx] == 0:
                        valid = False
                        break
                if valid:
                    return True

            # --- Vertical placements ---
            start_r_min = max(0, r - size + 1)
            start_r_max = min(r, board_size - size)
            for start_r in range(start_r_min, start_r_max + 1):
                valid = True
                # Check the vertical segment in column c from start_r to start_r+size-1.
                for i in range(start_r, start_r + size):
                    idx = i * board_size + c
                    if board[2][idx] == 1 or board[3][idx] == 1:
                        valid = False
                        break
                    if board[0][idx] == 1 and board[1][idx] == 0:
                        valid = False
                        break
                if valid:
                    return True

        # If no valid placement exists for any ship size, then prune the move.
        return False
