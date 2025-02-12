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
    plt.figure(figsize=(10, 5))
    plt.plot(move_count, label="Moves per game")
    plt.xlabel("Game")
    plt.ylabel("Moves")
    plt.title("Moves per game")
    plt.legend()
    plt.show()


def plot_action_distribution(action_distribution, board_size):
    """
    Prints the action distribution in a readable grid format with 3 decimal precision and colors.

    :param action_distribution: 1D NumPy array containing the probability of each action.
    :param board_size: The size of the Battleship board (assumed to be square).
    """
    action_grid = np.array(action_distribution).reshape((board_size, board_size))

    # Define color map (ANSI escape codes)
    def get_colored_value(value):
        if value > 0.05:
            return f"\033[91m{value:.3f}\033[0m"  # Red for high probability
        elif value > 0.03:
            return f"\033[93m{value:.3f}\033[0m"  # Yellow for medium probability
        elif value > 0.01:
            return f"\033[92m{value:.3f}\033[0m"  # Green for low probability
        else:
            return f"\033[90m{value:.3f}\033[0m"  # Grey for very low probability

    print("\nMCTS Action Distribution:")
    print("-" * (board_size * 7))  # Formatting line

    for row in range(board_size):
        row_values = [
            get_colored_value(action_grid[row, col]) for col in range(board_size)
        ]
        print(" | ".join(row_values))  # Print row with values separated by "|"

    print("-" * (board_size * 7))  # Formatting line


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
