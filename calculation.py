import matplotlib.pyplot as plt
from itertools import product
import time


def valid_ship_placements(board_size, ship_sizes):
    """
    Calculate the total number of valid ship placements on a board (exact recursive calculation).
    """
    rows, cols = board_size
    total_configurations = 0

    def is_valid_placement(board, ship, start, direction):
        r, c = start
        for i in range(ship):
            if direction == "H":  # Horizontal
                if c + i >= cols or board[r][c + i] != 0:
                    return False
            elif direction == "V":  # Vertical
                if r + i >= rows or board[r + i][c] != 0:
                    return False
        return True

    def place_ship(board, ship, start, direction):
        r, c = start
        for i in range(ship):
            if direction == "H":
                board[r][c + i] = 1
            elif direction == "V":
                board[r + i][c] = 1

    def remove_ship(board, ship, start, direction):
        r, c = start
        for i in range(ship):
            if direction == "H":
                board[r][c + i] = 0
            elif direction == "V":
                board[r + i][c] = 0

    def backtrack(board, ship_index):
        nonlocal total_configurations
        if ship_index == len(ship_sizes):  # All ships placed
            total_configurations += 1
            return
        ship = ship_sizes[ship_index]
        for r, c in product(range(rows), range(cols)):
            for direction in ["H", "V"]:
                if is_valid_placement(board, ship, (r, c), direction):
                    place_ship(board, ship, (r, c), direction)
                    backtrack(board, ship_index + 1)
                    remove_ship(board, ship, (r, c), direction)

    board = [[0 for _ in range(cols)] for _ in range(rows)]
    print(f"Calculating exact placements for ship sizes: {ship_sizes} on board {rows}x{cols}...")
    start_time = time.time()
    backtrack(board, 0)
    end_time = time.time()
    print(f"Done! Time taken: {end_time - start_time:.2f} seconds. Total configurations: {total_configurations}")
    return total_configurations


def approximate_ship_placements(board_size, ship_sizes):
    """
    Approximate the number of valid ship placements on a board.
    """
    rows, cols = board_size
    total_positions = 1
    available_cells = rows * cols

    for ship_size in ship_sizes:
        # Calculate horizontal and vertical placements for this ship
        horizontal_positions = rows * (cols - ship_size + 1) if cols - ship_size + 1 > 0 else 0
        vertical_positions = (rows - ship_size + 1) * cols if rows - ship_size + 1 > 0 else 0
        ship_placements = horizontal_positions + vertical_positions

        # Apply refined overlap adjustment
        blocking_factor = (available_cells - 2 * ship_size) / available_cells
        blocking_factor = max(0.1, blocking_factor)  # Prevent extreme reduction to zero
        total_positions *= ship_placements * blocking_factor

        # Reduce available cells dynamically
        available_cells -= ship_size

    print(f"Approximation for ship sizes {ship_sizes} on board {rows}x{cols}: {total_positions:.2f}")

    return total_positions


def generate_combined_plot():
    """
    Generate a graph comparing exact and approximate calculations of ship placements.
    """
    board_sizes = [(5, 5)]
    ship_sizes_sets = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]

    # Store results
    exact_results = {}
    approximate_results = {}

    for board_size in board_sizes:
        exact_results[board_size] = []
        approximate_results[board_size] = []
        for ship_sizes in ship_sizes_sets:
            # Run exact calculation
            exact_count = valid_ship_placements(board_size, ship_sizes)
            exact_results[board_size].append(exact_count)

            # Run approximation
            approx_count = approximate_ship_placements(board_size, ship_sizes)
            approximate_results[board_size].append(approx_count)

    # Plot the results
    plt.figure(figsize=(12, 8))
    for board_size in board_sizes:
        labels = ["-".join(map(str, sizes)) for sizes in ship_sizes_sets]
        plt.plot(labels, exact_results[board_size], marker="o", label=f"Exact (Board: {board_size[0]}x{board_size[1]})")
        plt.plot(labels, approximate_results[board_size], marker="x", linestyle="--",
                 label=f"Approximation (Board: {board_size[0]}x{board_size[1]})")

    plt.title("Exact vs Approximate Ship Placements", fontsize=16)
    plt.xlabel("Ship Sizes", fontsize=14)
    plt.ylabel("Number of Configurations", fontsize=14)
    plt.legend(title="Calculation Type", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# Generate the combined graph
generate_combined_plot()
