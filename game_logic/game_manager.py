import numpy as np

from game_logic.game_state import GameState
import copy


class GameManager:
    """
    Class to manage the game state between a placing agent and a search agent
    Counts the number of moves and checks if the game is over
    """

    def __init__(self, size):
        self.size = size

    def initial_state(self, placing):
        board = [[0 for _ in range(self.size**2)] for _ in range(4)]

        return GameState(
            board=board,
            move_count=0,
            placing=placing,
            remaining_ships=placing.ship_sizes.copy(),
        )

    def get_legal_moves(self, state):
        legal_moves = []
        for i in range(self.size**2):
            if state.board[0][i] == 0:
                legal_moves.append(i)

        return legal_moves

    def next_state(self, state, move, sim_id=None):
        new_board = [row[:] for row in state.board]
        new_move_count = state.move_count
        new_board[0][move] = 1
        new_remaining_ships = state.remaining_ships.copy()

        if move in state.placing.indexes:
            new_board[1][move] = 1
            sunk, ship_size, hit_ship = self.check_ship_sunk(
                move, new_board, state.placing
            )

            if sunk:
                if hit_ship is None:
                    print("Warning: hit_ship is None")
                if ship_size in new_remaining_ships:
                    for i in hit_ship:
                        new_board[3][i] = 1
                    new_remaining_ships.remove(ship_size)
                else:
                    print(
                        "Warning: sunk ship size",
                        ship_size,
                        "not found in remaining_ships",
                    )
        else:
            new_board[2][move] = 1
        new_move_count += 1
        return GameState(new_board, new_move_count, state.placing, new_remaining_ships)

    def check_ship_sunk(self, move, board, placing):
        hit_ship = None
        sunk = True
        # Find the ship that was hit
        for ship in placing.ships:
            if move in ship.indexes:
                hit_ship = ship.indexes

        sunk = all(board[1][i] == 1 for i in hit_ship)

        # Return if the ship is sunk and the ship size
        return sunk, len(hit_ship), hit_ship

    def is_terminal(self, state):
        return len(state.remaining_ships) == 0

    def simulate_game(self, placing_agent, search_agent):
        """
        Plays one game with the given search agent and returns a tuple of metrics:
          (total_moves,
           hit_accuracy,
           avg_sink_efficiency,
           avg_moves_between_hits,
           start_entropy,
           end_entropy)
        """
        current_state = self.initial_state(placing=placing_agent)

        # Initialize metrics
        total_moves = 0
        hits = 0
        misses = 0
        last_hit_move = None
        moves_between_hits = []
        ship_hit_times = {}
        sink_moves = []
        entropy_distributions = []

        while not self.is_terminal(current_state):
            # get move + (optional) probability distribution
            result = search_agent.strategy.find_move(current_state)
            if isinstance(result, tuple):
                move, prob_np = result
            else:
                move = result
                # uniform fallback
                prob_np = np.ones(self.size**2) / (self.size**2)

            # mask out illegal moves
            flat_board = np.array(current_state.board[0]).flatten()
            legal_mask = (flat_board == 0)
            entropy_distributions.append(prob_np[legal_mask])

            # check hit
            is_hit = move in current_state.placing.indexes

            # Process the move
            new_state = self.next_state(current_state, move)

            if is_hit:
                hits += 1
                if last_hit_move is not None:
                    moves_between_hits.append(total_moves - last_hit_move)
                last_hit_move = total_moves

                # check for sinking
                sunk, ship_size, hit_ship = self.check_ship_sunk(
                    move, new_state.board, current_state.placing
                )
                if hit_ship:
                    ship_id = tuple(sorted(hit_ship))
                    if ship_id not in ship_hit_times:
                        ship_hit_times[ship_id] = total_moves
                    if sunk:
                        sink_moves.append(total_moves - ship_hit_times[ship_id] + 1)
            else:
                misses += 1

            # advance state
            current_state = new_state
            total_moves += 1

        # --- compute final metrics ---
        accuracy = hits / total_moves if total_moves > 0 else 0.0
        avg_sink_eff = sum(sink_moves) / len(sink_moves) if sink_moves else 0.0
        avg_moves_btwn = (
            sum(moves_between_hits) / len(moves_between_hits)
            if moves_between_hits else 0.0
        )

        # entropy: mean of first/last 3 moves
        entropies = [self.calculate_entropy(d) for d in entropy_distributions]
        if len(entropies) >= 3:
            start_ent = float(np.mean(entropies[:3]))
            end_ent   = float(np.mean(entropies[-3:]))
        else:
            start_ent = end_ent = float(np.mean(entropies)) if entropies else 0.0

        return (
            total_moves,
            accuracy,
            avg_sink_eff,
            avg_moves_btwn,
            start_ent,
            end_ent
        )

    def calculate_entropy(self, distribution):
        """
            Calculate the entropy of a probability distribution.
            Higher entropy means more uniform distribution (less concentrated).
            Lower entropy means more concentrated distribution (more certainty).

            :param distribution: A probability distribution (numpy array)
            :return: The entropy value
        """
        # Filter out zero probabilities to avoid log(0)
        probabilities = distribution[distribution > 0]

        if len(probabilities) == 0:
            return 0

        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(distribution))
        if max_entropy == 0:
            return 0
        normalized_entropy = entropy / max_entropy

        return normalized_entropy
