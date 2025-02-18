import random

def is_gene_valid(board, col, row, direction, board_size, size):
    """
    Check if placing a ship with the given gene is valid on the board.

    Parameters:
      - board: 2D list (rows × columns) with 0 for empty and 1 for occupied.
      - col, row: Starting column and row.
      - direction: 0 for horizontal, 1 for vertical.
      - board_size: Size of the (square) board.
      - size: Length of the ship.

    Returns True if the ship fits on the board and does not overlap; otherwise False.
    """
    if direction == 0:  # horizontal: ship extends rightwards
        if col + size > board_size:
            return False
        for j in range(size):
            if board[row][col + j] == 1:
                return False
    else:  # vertical: ship extends downwards
        if row + size > board_size:
            return False
        for j in range(size):
            if board[row + j][col] == 1:
                return False
    return True


def mark_board(board, gene, board_size, size):
    """
    Mark the board cells as occupied by a ship defined by the gene.

    Parameters:
      - board: 2D list representing the board.
      - gene: (col, row, direction).
      - board_size: Size of the board.
      - size: Length of the ship.
    """
    col, row, direction = gene
    if direction == 0:  # horizontal
        for j in range(size):
            board[row][col + j] = 1
    else:  # vertical
        for j in range(size):
            board[row + j][col] = 1


def random_valid_gene(board, board_size, size):
    """
    Generate a random gene (col, row, direction) that is valid on the board.

    Parameters:
      - board: 2D list representing current board occupancy.
      - board_size: Size of the board.
      - size: Length of the ship.

    Returns a valid gene.
    """
    while True:
        direction = random.randint(0, 1)
        if direction == 0:  # horizontal
            col = random.randint(0, board_size - size)
            row = random.randint(0, board_size - 1)
        else:  # vertical
            col = random.randint(0, board_size - 1)
            row = random.randint(0, board_size - size)
        if is_gene_valid(board, col, row, direction, board_size, size):
            return (col, row, direction)


def local_mutation_gene(gene, board, board_size, size):
    """
    Try to perform a local mutation (small shift or toggle direction) on a gene.
    Returns a mutated gene that is valid.

    Parameters:
      - gene: The original gene (col, row, direction).
      - board: 2D list representing current placements.
      - board_size: Size of the board.
      - size: Length of the ship.
    """
    col, row, direction = gene
    candidates = []
    # Try small shifts in col and row by -1, 0, or +1.
    for dcol in [-1, 0, 1]:
        for drow in [-1, 0, 1]:
            new_col = col + dcol
            new_row = row + drow
            # Check bounds for the current direction.
            if direction == 0:  # horizontal
                if new_col < 0 or new_col >= board_size or new_row < 0 or new_row > board_size - size:
                    continue
            else:  # vertical
                if new_col < 0 or new_col > board_size - size or new_row < 0 or new_row >= board_size:
                    continue
            candidates.append((new_col, new_row, direction))
    # Also try toggling the direction (keeping col, row)
    new_direction = 1 - direction
    if new_direction == 0:  # horizontal
        if col >= 0 and col < board_size and row >= 0 and row <= board_size - size:
            candidates.append((col, row, new_direction))
    else:  # vertical
        if col >= 0 and col <= board_size - size and row >= 0 and row < board_size:
            candidates.append((col, row, new_direction))
    random.shuffle(candidates)
    for candidate in candidates:
        if is_gene_valid(board, candidate[0], candidate[1], candidate[2], board_size, size):
            return candidate
    # Fallback: if no candidate is valid, generate a random valid gene.
    return random_valid_gene(board, board_size, size)
