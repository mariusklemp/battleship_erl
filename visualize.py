import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_stats(statistics, ylog=False, view=False, filename="avg_fitness.svg"):
    """Plots the population's average and best fitness."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, "b-", label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, "g-.", label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """Plots the trains for a single spiking neuron."""
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename="speciation.svg"):
    """Visualizes speciation throughout evolution."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    if view:
        plt.show()

    plt.close()


def draw_net(
        config,
        genome,
        view=False,
        filename=None,
        node_names=None,
        show_disabled=True,
        prune_unused=False,
        node_colors=None,
        fmt="svg",
):
    """Receives a genome and draws a neural network with arbitrary topology."""
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)"
        )
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {"shape": "circle", "fontsize": "9", "height": "0.2", "width": "0.2"}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {
            "style": "filled",
            "shape": "box",
            "fillcolor": node_colors.get(k, "lightgray"),
        }
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "green" if cg.weight > 0 else "red"
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(
                a, b, _attributes={"style": style, "color": color, "penwidth": width}
            )

    dot.render(filename, view=view)

    return dot


""" A function to plot the move count """


def plot_fitness(move_count, board_size):
    """
    Plot the move count evolution over games, including statistics and a moving average.

    Args:
        move_count (list): List of moves taken in each game
        board_size (int): Size of the game board
    """
    # Convert to numpy array for easier manipulation
    moves = np.array(move_count)
    games = np.arange(len(moves))

    # Calculate statistics
    moving_avg_window = min(20, len(moves))  # Adaptive window size
    moving_avg = np.convolve(
        moves, np.ones(moving_avg_window) / moving_avg_window, mode="valid"
    )
    overall_avg = np.mean(moves)
    best_score = np.min(moves)
    worst_score = np.max(moves)

    # Create the plot with a modern style
    plt.style.use("bmh")  # Using built-in style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.patch.set_facecolor("white")  # Set background color

    # Set title with custom styling
    fig.suptitle(
        f"Battleship Performance Analysis (Board Size: {board_size}x{board_size})",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    # Main plot
    ax1.plot(
        games, moves, color="#E6E6E6", label="Moves per game", alpha=0.5, linewidth=1
    )
    ax1.plot(
        games[moving_avg_window - 1:],
        moving_avg,
        color="#2E86C1",
        label=f"Moving average (window={moving_avg_window})",
        linewidth=2,
    )
    ax1.axhline(
        y=overall_avg,
        color="#E74C3C",
        linestyle="--",
        label=f"Overall average: {overall_avg:.1f}",
        linewidth=1.5,
    )

    # Add min/max bands
    ax1.fill_between(
        games,
        np.minimum.accumulate(moves),
        np.maximum.accumulate(moves),
        alpha=0.1,
        color="#7F8C8D",
        label="Min-Max Range",
    )

    # Formatting for main plot
    ax1.set_xlabel("Game Number", fontsize=10)
    ax1.set_ylabel("Number of Moves", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Statistics subplot
    stats_data = [
        ("Best Score", best_score),
        ("Worst Score", worst_score),
        ("Average", overall_avg),
        ("Last 20 Avg", np.mean(moves[-20:]) if len(moves) >= 20 else np.mean(moves)),
    ]

    # Create bar plot for statistics with custom colors
    colors = ["#2ECC71", "#E74C3C", "#3498DB", "#9B59B6"]
    bars = ax2.bar(
        range(len(stats_data)), [x[1] for x in stats_data], color=colors, alpha=0.7
    )
    ax2.set_xticks(range(len(stats_data)))
    ax2.set_xticklabels([x[0] for x in stats_data], fontsize=9)

    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Formatting for stats plot
    ax2.set_ylabel("Moves", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_action_distribution(action_distribution, board_size):
    """
    Prints the action distribution in a readable grid format with 3 decimal precision and dynamic colors.
    This version uses additional thresholds for a more detailed "heatmap" effect.

    :param action_distribution: 1D NumPy array containing the probability of each action.
    :param board_size: The size of the Battleship board (assumed to be square).
    """
    # Reshape the distribution into a grid.
    action_grid = np.array(action_distribution).reshape((board_size, board_size))

    # Compute the maximum probability value.
    max_val = np.max(action_grid)

    # Define dynamic thresholds as fractions of the maximum value.
    # You can adjust these fractions to change the color boundaries.
    threshold_very_high = 0.9 * max_val
    threshold_high = 0.75 * max_val
    threshold_med_high = 0.6 * max_val
    threshold_med = 0.45 * max_val
    threshold_med_low = 0.3 * max_val
    threshold_low = 0.15 * max_val

    def get_colored_value(value):
        if value >= threshold_very_high:
            return f"\033[95m{value:.3f}\033[0m"  # Magenta for very high values
        elif value >= threshold_high:
            return f"\033[91m{value:.3f}\033[0m"  # Red for high values
        elif value >= threshold_med_high:
            return f"\033[93m{value:.3f}\033[0m"  # Yellow for medium-high values
        elif value >= threshold_med:
            return f"\033[92m{value:.3f}\033[0m"  # Green for medium values
        elif value >= threshold_med_low:
            return f"\033[96m{value:.3f}\033[0m"  # Cyan for medium-low values
        elif value >= threshold_low:
            return f"\033[94m{value:.3f}\033[0m"  # Blue for low values
        else:
            return f"\033[90m{value:.3f}\033[0m"  # Grey for very low values

    print("\nMCTS Action Distribution:")
    print("-" * (board_size * 7))  # Separator line

    for row in range(board_size):
        row_values = [get_colored_value(action_grid[row, col]) for col in range(board_size)]
        print(" | ".join(row_values))

    print("-" * (board_size * 7))  # Separator line


def show_board(board, board_size):
    """
    Prints the Battleship board with colors for better visualization.

    :param state: The game state containing the board layers.
    :param board_size: The size of the board (assumed square).
    """
    # ANSI Color Codes
    COLORS = {
        "empty": "\033[90m- \033[0m",  # Grey for unexplored
        "hit": "\033[93mX \033[0m",  # Yellow for hits
        "miss": "\033[94mO \033[0m",  # Blue for misses
        "sunk": "\033[91mS \033[0m",  # Red for sunken ships
    }

    print("\nCurrent Board State:")
    print("-" * (board_size * 2))  # Formatting line

    for i in range(board_size):
        row = ""
        for j in range(board_size):
            index = i * board_size + j
            if board[3][index] == 1:  # Sunken ship
                row += COLORS["sunk"]
            elif board[1][index] == 1:  # Hit
                row += COLORS["hit"]
            elif board[2][index] == 1:  # Miss
                row += COLORS["miss"]
            else:  # Unexplored
                row += COLORS["empty"]

        print(row)

    print("-" * (board_size * 2))  # Formatting line


import numpy as np


def print_rbuf(rbuf, num_samples, board_size):
    """
    Pretty-print a few samples from the replay buffer.

    :param rbuf: The replay buffer object.
    :param num_samples: The number of samples to print (default: 5).
    """
    print("\n--- Replay Buffer Contents ---")
    print(f"Total stored samples: {len(rbuf.data)}")

    if len(rbuf.data) == 0:
        print("Replay buffer is empty.")
        return

    # Randomly sample entries to print if buffer is large
    indices = np.random.choice(
        len(rbuf.data), min(num_samples, len(rbuf.data)), replace=False
    )

    for idx in indices:
        (board_tensor, extra_features), action_distribution = rbuf.data[idx]

        print(f"\nSample {idx + 1}:")
        show_board(board_tensor.numpy().reshape(-1, board_tensor.shape[-1]), board_size)
        print(
            "Extra Features:",
            (
                extra_features.numpy()
                if isinstance(extra_features, torch.Tensor)
                else extra_features
            ),
        )
        # plot_action_distribution(action_distribution, board_size)
        print("-" * 40)
