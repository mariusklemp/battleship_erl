from functools import lru_cache


def generate_bitmask_placements(size, N):
    """
    Return a list of *unique* bitmasks for all ways to place
    a ship of given length on an N×N board, horizontal or vertical.
    Bits are laid out row-major: bit (r*N + c).
    """
    masks = set()

    # Horizontal placements
    for r in range(N):
        for c in range(N - size + 1):
            m = 0
            for i in range(size):
                m |= 1 << (r * N + (c + i))
            masks.add(m)

    # Vertical placements
    for c in range(N):
        for r in range(N - size + 1):
            m = 0
            for i in range(size):
                m |= 1 << ((r + i) * N + c)
            masks.add(m)

    return list(masks)


def count_configurations(N, ship_sizes):
    """
    Count non‐overlapping ways to place ships of lengths in ship_sizes
    on an N×N board, treating equal‐length ships as identical.
    """
    # sort descending so bigger ships place first (pruning faster)
    ships = tuple(sorted(ship_sizes, reverse=True))
    # pre‐compute placement lists for each distinct length
    placement_lists = {
        s: generate_bitmask_placements(s, N)
        for s in set(ships)
    }

    @lru_cache(None)
    def backtrack(idx, occupied, last_choice):
        """
        idx: which ship we’re placing (0..len(ships))
        occupied: bitmask of already‐used cells
        last_choice: index in placement_list[ships[idx]] used by the _previous_
                     ship of the *same* length, or -1 if none
        """
        if idx == len(ships):
            return 1

        length = ships[idx]
        placements = placement_lists[length]
        total = 0

        # Determine where to start so identical ships don't reorder:
        start = 0
        if idx > 0 and ships[idx] == ships[idx - 1]:
            start = last_choice + 1

        for choice in range(start, len(placements)):
            mask = placements[choice]
            if (mask & occupied) == 0:
                # pass new last_choice for this ship if it matches next
                next_last = choice
                total += backtrack(
                    idx + 1,
                    occupied | mask,
                    next_last
                )

        return total

    return backtrack(0, 0, -1)


# Examples:
print(count_configurations(5, [3, 3, 2]))  # → 8000
print(count_configurations(6, [3, 3, 2]))  # → 40324
print(count_configurations(7, [3, 2, 2]))  # → 182 938
print(count_configurations(7, [4, 3, 2, 2]))  # → 6 078 844
print(count_configurations(3, [2, 2]))  # → 44
print(count_configurations(3, [3, 2]))  # → 36
print(count_configurations(3, [3, 3]))  # → 6
print(count_configurations(3, [1]))  # → 9
