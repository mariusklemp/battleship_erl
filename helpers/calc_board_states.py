import matplotlib.pyplot as plt
from functools import lru_cache

from matplotlib import rcParams


def generate_bitmask_placements(size, N):
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
    print(f"Calculating configurations for board size {N} with ships {ship_sizes}")
    ships = tuple(sorted(ship_sizes, reverse=True))
    placement_lists = {
        s: generate_bitmask_placements(s, N)
        for s in set(ships)
    }

    @lru_cache(None)
    def backtrack(idx, occupied, last_choice):
        if idx == len(ships):
            return 1

        length = ships[idx]
        placements = placement_lists[length]
        total = 0

        start = 0
        if idx > 0 and ships[idx] == ships[idx - 1]:
            start = last_choice + 1

        for choice in range(start, len(placements)):
            mask = placements[choice]
            if (mask & occupied) == 0:
                total += backtrack(
                    idx + 1,
                    occupied | mask,
                    choice
                )

        return total

    return backtrack(0, 0, -1)


# Example usage
# print result from max board size 10 and full ships [5,4,3,2,2]
# print(count_configurations(10, [5, 4, 3, 3, 2]))
# print(count_configurations(7, [3, 3, 2]))
# print(count_configurations(5, [3, 3, 2]))

# Plot 1: Effect of board size (fixed ship sizes)
print("Effect of board size on configurations")
ship_sizes = [3, 3, 2]
sizes = list(range(3, 11))  # Board sizes 4x4 to 8x8

config_counts_by_board = []
for N in sizes:
    res = count_configurations(N, ship_sizes)
    config_counts_by_board.append(res)
    print("res", res)

print("Effect of ship loadout on configurations")
# Plot 2: Effect of ship loadout (fixed board size)
board_size = 10
ship_variants = [
    [5, 4, 3, 2, 2],
]
labels = [str(s) for s in ship_variants]

config_counts_by_ships = [count_configurations(board_size, s) for s in ship_variants]


plt.figure(figsize=(12, 5))

# Plot 1 — Board size vs configurations
plt.subplot(1, 2, 1)
plt.plot(sizes, config_counts_by_board, marker='o', linewidth=2, markersize=6)
plt.yscale('log')
plt.title('Board Size vs. Configurations')
plt.xlabel('Board Size (N × N)')
plt.ylabel('Valid Configurations')
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)

# Plot 2 — Ship loadout vs configurations
plt.subplot(1, 2, 2)
bars = plt.bar(labels, config_counts_by_ships)
plt.yscale('log')
plt.title(f'Ship Loadout vs. Configurations (Board: {board_size}×{board_size})')
plt.xlabel('Ship Sizes')
plt.ylabel('Valid Configurations')
plt.xticks(rotation=30)
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)

# Tight layout for better alignment
plt.tight_layout()
plt.show()
