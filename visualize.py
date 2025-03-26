import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from neat_system.cnn_layers import CNNConvGene, CNNPoolGene, CNNFCGene


def plot_innovation_registry(innovation_registry):
    """
    Plots the global innovation registry sorted by innovation number.
    Each innovation is represented as a box (with up to 10 per row) showing:
      - Innovation number (in parentheses)
      - Gene type
      - Signature details
    """
    # Sort innovations by their number.
    sorted_items = sorted(innovation_registry.items(), key=lambda item: item[1])
    n = len(sorted_items)
    max_cols = 10
    nrows = (n + max_cols - 1) // max_cols

    fig, ax = plt.subplots(figsize=(1.5 * max_cols, 3 * nrows))

    # Define colors for each gene type.
    color_map = {
        "CNNConvGene": "#c2f0c2",  # pale green
        "CNNPoolGene": "#c2e0f0",  # pale blue
        "CNNFCGene": "#f0d9c2",  # pale orange
    }

    for i, ((gene_type, signature), innov_number) in enumerate(sorted_items):
        col = i % max_cols
        row = i // max_cols
        # Plot rows from top to bottom.
        y = nrows - row - 1
        line1 = f"({innov_number})"
        line2 = f"{gene_type}"
        line3 = " | ".join(str(x) for x in signature)
        text_str = f"{line1}\n{line2}\n{line3}"
        face_color = color_map.get(gene_type, "#dddddd")
        rect = Rectangle((col, y), 0.9, 1, facecolor=face_color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(col + 0.45, y + 0.5, text_str, ha="center", va="center",
                wrap=True, fontsize=9)

    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title("Global Innovation Registry")
    plt.tight_layout()
    plt.show()


def plot_genome_simple(genome, label="", max_cols=5, fig_size=(8, 4)):
    """
    Creates a new figure and axes, then plots the specified genome
    using plot_genome. Finally, shows the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    plot_genome(ax, genome, label, max_cols=max_cols)
    plt.show()


def plot_genome(ax, genome, label, max_cols=5):
    """
    Plots the layers (genes) of a single genome in a fixed grid,
    ensuring each gene box has the same size as in the innovation registry.
    """
    n = len(genome.layer_config)
    nrows = (n + max_cols - 1) // max_cols  # total rows needed

    color_map = {
        "CNNConvGene": "#c2f0c2",  # pale green
        "CNNPoolGene": "#c2e0f0",  # pale blue
        "CNNFCGene": "#f0d9c2",  # pale orange
    }

    for i, gene in enumerate(genome.layer_config):
        # Column and row within this grid
        col = i % max_cols
        row = i // max_cols

        # We plot from top (row=0) to bottom (row=nrows-1),
        # so we invert row by subtracting from nrows-1:
        y = (nrows - 1) - row
        x = col

        gtype = gene.__class__.__name__ if hasattr(gene, "__class__") else "Unknown"
        if gtype == "CNNConvGene":
            gene_type = f"Conv {gene.key}"
            details1 = f"k={gene.kernel_size}, s={gene.stride}, p={gene.padding}"
            details2 = f"out={gene.out_channels}"
        elif gtype == "CNNPoolGene":
            gene_type = f"Pool {gene.key}"
            details1 = f"size={gene.pool_size}, s={gene.stride}"
            details2 = f"type={gene.pool_type}"
        elif gtype == "CNNFCGene":
            gene_type = f"FC {gene.key}"
            details1 = f"size={gene.fc_layer_size}"
            details2 = f"act={gene.activation}"
        else:
            gene_type = f"Unknown {gene.key}"
            details1 = "???"
            details2 = ""

        status = "enabled" if gene.enabled else "disabled"
        face_color = color_map.get(gtype, "#dddddd") if gene.enabled else "#dddddd"

        # Each box is 0.9 wide and 1.0 tall, just like in your registry.
        rect = Rectangle((x, y), 0.9, 1, facecolor=face_color, edgecolor="black")
        ax.add_patch(rect)
        text_str = f"{gene_type}\n{details1}\n{details2}\n{status}"
        ax.text(x + 0.45, y + 0.5, text_str, ha="center", va="center",
                wrap=True, fontsize=9)

    # Force the axis to span exactly from 0..max_cols in X and 0..nrows in Y.
    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, nrows)

    # Keep each box square. This ensures the boxes remain the same physical size.
    ax.set_aspect("equal", adjustable="box")

    # Hide the axis lines/ticks.
    ax.axis("off")

    # Title for this genome.
    ax.set_title(f"Genome ({genome.key}): {label}")


def plot_multiple_genomes(genome_label_list, fig_title, max_cols=8):
    """
    Plots several genomes in subplots arranged vertically (one per row).
    Each genome uses the same 'fixed-size' box approach.
    """
    n_genomes = len(genome_label_list)
    nrows = n_genomes
    ncols = 1

    # Increase width & height to give enough space.
    # The width of 1.5*max_cols matches your registry style;
    # multiply the height to accommodate multiple genomes.
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * max_cols, 3 * n_genomes))

    # If only one genome, make `axes` iterable.
    if n_genomes == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # Plot each genome in its own subplot row.
    for ax, (genome, label) in zip(axes, genome_label_list):
        plot_genome(ax, genome, label, max_cols=max_cols)

    plt.suptitle(fig_title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_species_weight_stats(species_weight_stats):
    """
    Plots the evolution of the mean weight value per species over generations.

    Parameters:
      species_weight_stats: A dict with keys as generations and values as dicts
                            mapping species IDs to weight statistics (including 'mean').
    """
    all_species = set()
    for gen_stats in species_weight_stats.values():
        all_species.update(gen_stats.keys())
    all_species = sorted(list(all_species))

    species_data = {sid: {} for sid in all_species}
    for gen, gen_stats in species_weight_stats.items():
        for sid in all_species:
            if sid in gen_stats:
                species_data[sid][gen] = gen_stats[sid]['mean']
            else:
                species_data[sid][gen] = np.nan

    df = pd.DataFrame({sid: pd.Series(data) for sid, data in species_data.items()})
    df.index.name = "Generation"

    plt.figure(figsize=(10, 6))
    for sid in df.columns:
        plt.plot(df.index, df[sid], marker='o', label=f"Species {sid}")
    plt.xlabel("Generation")
    plt.ylabel("Mean Weight Value")
    plt.title("Mean Weight Evolution Per Species")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weight_stats(weight_stats):
    """
    Plots the mean, std deviation, min, and max values of weights over generations.
    """
    df = pd.DataFrame(weight_stats).T
    df.index.name = "Generation"

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['mean'], marker='o', label='Mean Weight')
    plt.fill_between(df.index, df['min'], df['max'], color='gray', alpha=0.3, label='Min/Max Range')
    plt.plot(df.index, df['std'], marker='o', label='Weight Std Dev')

    plt.xlabel("Generation")
    plt.ylabel("Weight Values")
    plt.title("Weight Evolution Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitness_boxplot(stats):
    fitness_data = []
    generations = []
    for gen_idx, gen_stats in enumerate(stats.generation_statistics):
        gen_fitness = []
        for species in gen_stats.values():
            gen_fitness.extend(species.values())
        fitness_data.append(gen_fitness)
        generations.append(gen_idx)

    plt.figure(figsize=(12, 6))
    plt.boxplot(fitness_data, labels=generations, showfliers=True)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Distribution Per Generation")
    plt.grid(True)
    plt.show()


def plot_species_sizes(stats):
    species_counts = stats.get_species_sizes()
    species_df = pd.DataFrame(species_counts)
    species_df.columns = [f"Species {i + 1}" for i in range(species_df.shape[1])]
    species_df.index.name = "Generation"

    plt.figure(figsize=(10, 6))
    for col in species_df.columns:
        plt.plot(species_df.index, species_df[col], marker='o', label=col)
    plt.xlabel("Generation")
    plt.ylabel("Number of Members")
    plt.title("Species Sizes Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_hof(statistics, hof_size=5):
    best_genomes = statistics.best_genomes(hof_size)
    fig, axes = plt.subplots(1, hof_size, figsize=(5 * hof_size, 5))

    if hof_size == 1:
        axes = [axes]

    for idx, genome in enumerate(best_genomes):
        ax = axes[idx]
        visualize_genome(genome, ax, title=f"HOF Genome {idx + 1}")

    plt.tight_layout()
    plt.show()


def visualize_genome(genome, ax, title="Genome Architecture"):
    G = nx.DiGraph()
    # Only include active genes
    active_genes = [gene for gene in genome.layer_config if getattr(gene, "enabled", True)]
    for i, gene in enumerate(active_genes):
        if isinstance(gene, CNNConvGene):
            label = f"Conv: k={gene.kernel_size}\nout={gene.out_channels}"
        elif isinstance(gene, CNNPoolGene):
            label = f"Pool: type={gene.pool_type}\np={gene.pool_size}, s={gene.stride}"
        elif isinstance(gene, CNNFCGene):
            label = f"FC: size={gene.fc_layer_size}"
        else:
            label = "Unknown"
        G.add_node(i, label=label)
        if i > 0:
            G.add_edge(i - 1, i)

    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1500,
            node_color='lightblue', ax=ax)
    ax.set_title(title)


def visualize_genome_simple(genome, title="Genome Architecture"):
    """
    Creates a new figure and axes, then calls visualize_genome
    on the provided genome. Finally, shows the plot immediately.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    visualize_genome(genome, ax, title=title)
    plt.show()


def visualize_species(statistics):
    species_counts = statistics.get_species_sizes()
    species_df = pd.DataFrame(species_counts)
    species_df.columns = [f"Species {i + 1}" for i in range(species_df.shape[1])]
    species_df.index.name = "Generation"

    for col in species_df.columns:
        nonzero = species_df[col] > 0
        if nonzero.any():
            first_gen = species_df.index[nonzero.argmax()]
            last_gen = species_df.index[::-1][nonzero[::-1].argmax()]
            species_df.loc[species_df.index < first_gen, col] = np.nan
            species_df.loc[species_df.index > last_gen, col] = np.nan

    plt.figure(figsize=(10, 6))
    for col in species_df.columns:
        plt.plot(species_df.index, species_df[col], marker='o', label=col)
    plt.xlabel("Generation")
    plt.ylabel("Number of Members")
    plt.title("Species Sizes Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stats(statistics, best_possible, ylog=False, view=False):
    generations = list(range(len(statistics.most_fit_genomes)))

    avg_fitness_list = statistics.get_fitness_mean()
    median_fitness_list = statistics.get_fitness_median()
    stdev_fitness_list = statistics.get_fitness_stdev()

    avg_fitness = np.array(avg_fitness_list) if isinstance(avg_fitness_list, list) else np.array([avg_fitness_list])
    median_fitness = np.array(median_fitness_list) if isinstance(median_fitness_list, list) else np.array(
        [median_fitness_list])
    stdev_fitness = np.array(stdev_fitness_list) if isinstance(stdev_fitness_list, list) else np.array(
        [stdev_fitness_list])

    best_fitness = [genome.fitness for genome in statistics.most_fit_genomes]

    avg_best_species_fitness = []
    for gen_stats in statistics.generation_statistics:
        best_species_fitness = []
        for species_id, genomes_fitness_dict in gen_stats.items():
            if genomes_fitness_dict:
                best_fit = max(genomes_fitness_dict.values())
                best_species_fitness.append(best_fit)
        avg_best_species_fitness.append(np.mean(best_species_fitness) if best_species_fitness else 0)

    min_length = min(len(generations), len(avg_fitness), len(best_fitness), len(avg_best_species_fitness),
                     len(median_fitness), len(stdev_fitness))
    generations = generations[:min_length]
    avg_fitness = avg_fitness[:min_length]
    best_fitness = best_fitness[:min_length]
    avg_best_species_fitness = avg_best_species_fitness[:min_length]
    median_fitness = median_fitness[:min_length]
    stdev_fitness = stdev_fitness[:min_length]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, "b-", marker='o', label="Average Fitness")
    plt.plot(generations, best_fitness, "g-", marker='o', label="Best Fitness")
    plt.plot(generations, median_fitness, "c-", marker='o', label="Median Fitness")
    plt.plot(generations, stdev_fitness, "y-", marker='o', label="Fitness Std Dev")
    plt.plot(generations, avg_best_species_fitness, "m-", marker='o', label="Avg. Best Fitness per Species")
    plt.axhline(y=best_possible, color="r", linestyle="--", label="Best Possible Fitness")

    plt.title("Population's Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    if view:
        plt.show()
    plt.close()


def analyze_species_from_population(species):
    species_analysis = {}
    for species_id, species_obj in species.species.items():
        conv_counts = []
        pool_counts = []
        fc_counts = []
        weight_means = []
        for genome in species_obj.members.values():
            # Count only enabled genes.
            conv_layers = sum(1 for gene in genome.layer_config
                              if isinstance(gene, CNNConvGene) and getattr(gene, "enabled", True))
            pool_layers = sum(1 for gene in genome.layer_config
                              if isinstance(gene, CNNPoolGene) and getattr(gene, "enabled", True))
            fc_layers = sum(1 for gene in genome.layer_config
                            if isinstance(gene, CNNFCGene) and getattr(gene, "enabled", True))
            conv_counts.append(conv_layers)
            pool_counts.append(pool_layers)
            fc_counts.append(fc_layers)
            layer_weight_means = [
                gene.weights.mean() for gene in genome.layer_config
                if hasattr(gene, 'weights') and getattr(gene, "enabled", True)
            ]
            if layer_weight_means:
                weight_means.append(np.mean(layer_weight_means))
        species_analysis[species_id] = {
            'avg_conv_layers': np.mean(conv_counts) if conv_counts else 0,
            'avg_pool_layers': np.mean(pool_counts) if pool_counts else 0,
            'avg_fc_layers': np.mean(fc_counts) if fc_counts else 0,
            'avg_weight_mean': np.mean(weight_means) if weight_means else 0,
        }
    return species_analysis


def plot_species_analysis(species_analysis):
    species_ids = sorted(species_analysis.keys())
    avg_conv_layers = [species_analysis[sid]['avg_conv_layers'] for sid in species_ids]
    avg_pool_layers = [species_analysis[sid]['avg_pool_layers'] for sid in species_ids]
    avg_fc_layers = [species_analysis[sid]['avg_fc_layers'] for sid in species_ids]

    x = np.arange(len(species_ids))
    width = 0.6

    plt.figure(figsize=(10, 6))
    p1 = plt.bar(x, avg_conv_layers, width, label='Avg Conv Layers', color='skyblue')
    p2 = plt.bar(x, avg_pool_layers, width, bottom=avg_conv_layers, label='Avg Pool Layers', color='lightcoral')
    p3 = plt.bar(x, avg_fc_layers, width, bottom=np.array(avg_conv_layers) + np.array(avg_pool_layers),
                 label='Avg FC Layers', color='lightgreen')

    plt.xlabel("Species ID")
    plt.ylabel("Average Number of Layers")
    plt.title("Stacked Layers (Conv + Pool + FC) per Species")
    plt.xticks(x, [f"Species {sid}" for sid in species_ids])
    plt.legend()
    plt.tight_layout()
    plt.show()


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
        games, moves, color="#FFFF00", label="Moves per game", alpha=0.5, linewidth=1
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
        row_values = [
            get_colored_value(action_grid[row, col]) for col in range(board_size)
        ]
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
    :param num_samples: The number of samples to print.
    :param board_size: The size of the board (assumed square).
    """
    print("\n--- Replay Buffer Contents ---")
    print(f"Total stored samples: {len(rbuf)}")

    if len(rbuf) == 0:
        print("Replay buffer is empty.")
        return

    # Randomly sample entries to print if buffer is large
    indices = np.random.choice(len(rbuf), min(num_samples, len(rbuf)), replace=False)

    for idx in indices:
        (board_tensor, extra_features), action_distribution = rbuf[idx]

        # Convert board tensor to numpy and reshape to the correct format
        # From (1, 4, board_size, board_size) to (4, board_size * board_size)
        board_array = board_tensor.squeeze(0).numpy()  # Remove batch dimension
        board_array = board_array.reshape(
            4, -1
        )  # Reshape to (4, board_size * board_size)

        print(f"\nSample {idx + 1}:")
        print("Board Shape:", board_tensor.shape)
        show_board(board_array, board_size)

        print(
            "Extra Features:",
            (
                extra_features.numpy()
                if isinstance(extra_features, torch.Tensor)
                else extra_features
            ),
        )
        print("Action Distribution Shape:", action_distribution.shape)
        plot_action_distribution(action_distribution, board_size)
        print("-" * 40)
