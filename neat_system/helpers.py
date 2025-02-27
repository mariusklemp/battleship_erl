from random import choice, randint

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis


# ============================================================================
# Helper functions to mutate weights
# ============================================================================

def _mutate_weights_conv(conv_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                         weight_min_value, weight_max_value, config):
    # Mutate weights and biases using the helper.
    conv_gene.weights = _mutate_array(conv_gene.weights, weight_mutate_rate, weight_mutate_power,
                                      weight_replace_rate, weight_min_value, weight_max_value, config)
    conv_gene.biases = _mutate_array(conv_gene.biases, weight_mutate_rate, weight_mutate_power,
                                     weight_replace_rate, weight_min_value, weight_max_value, config)


def _mutate_weights_fc(fc_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                       weight_min_value, weight_max_value, config):
    # Mutate weights and biases using the same helper.
    fc_gene.weights = _mutate_array(fc_gene.weights, weight_mutate_rate, weight_mutate_power,
                                    weight_replace_rate, weight_min_value, weight_max_value, config)
    fc_gene.biases = _mutate_array(fc_gene.biases, weight_mutate_rate, weight_mutate_power,
                                   weight_replace_rate, weight_min_value, weight_max_value, config)


# ============================================================================
# Helper functions to add new layers to a genome.
# ============================================================================

def calculate_conv_output_size(input_size, conv_gene):
    """Calculate the output size after a conv layer."""
    kernel_size = conv_gene.kernel_size
    stride = conv_gene.stride
    padding = conv_gene.padding
    output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1

    return output_size


def calculate_pool_output_size(input_size, pool_gene):
    """Calculate the output size after a pooling layer."""
    pool_size = pool_gene.pool_size
    stride = pool_gene.stride

    output_size = ((input_size - pool_size) // stride) + 1
    return max(1, output_size)  # Ensure valid size


# ============================================================================
# Helper functions to adapt weights when dimensions change.
# ============================================================================
def _mutate_array(arr, mutate_rate, mutate_power, replace_rate, weight_min, weight_max, config):
    """
    Applies controlled noise and selective replacement to a NumPy array (weights or biases).

    Parameters:
      arr: The original NumPy array.
      mutate_rate: Probability of applying additive noise.
      mutate_power: Standard deviation of the noise.
      replace_rate: Probability of fully replacing an element.
      weight_min, weight_max: Clipping bounds.
      config: Configuration object (to decide the initialization type).

    Returns:
      The mutated array.
    """
    # --- Additive Noise Mutation ---
    mask = np.random.rand(*arr.shape) < mutate_rate
    # Generate noise and clip it to avoid extreme values.
    noise = np.clip(np.random.randn(*arr.shape) * mutate_power, -mutate_power * 2, mutate_power * 2)
    arr = arr + mask * noise

    # --- Replacement Mutation (Less Frequent) ---
    rep_mask = np.random.rand(*arr.shape) < replace_rate
    new_vals = np.random.randn(*arr.shape) * config.weight_init_stdev + config.weight_init_mean

    # Clip the new values and apply replacement.
    new_vals = np.clip(new_vals, weight_min, weight_max)
    arr = np.where(rep_mask, new_vals, arr)

    # Ensure the final array is within the allowed range.
    return np.clip(arr, weight_min, weight_max)


def adapt_conv_weights(old_weights, new_shape):
    """
    Adapt old convolution weights (old_shape: (old_out, old_in, old_k, old_k))
    to new_shape (new_out, new_in, new_k, new_k) by copying the overlapping
    region. For kernel dimensions, we center-crop (or pad) the kernels.
    """
    new_weights = np.random.randn(*new_shape) * 0.1  # new random init
    old_out, old_in, old_k, _ = old_weights.shape
    new_out, new_in, new_k, _ = new_shape

    # Overlap for channels:
    out_overlap = min(old_out, new_out)
    in_overlap = min(old_in, new_in)

    # For the kernel dimensions, we try to center the old kernel inside the new one.
    kernel_overlap = min(old_k, new_k)
    old_start = (old_k - kernel_overlap) // 2
    new_start = (new_k - kernel_overlap) // 2

    new_weights[:out_overlap, :in_overlap, new_start:new_start + kernel_overlap, new_start:new_start + kernel_overlap] = \
        old_weights[:out_overlap, :in_overlap, old_start:old_start + kernel_overlap,
        old_start:old_start + kernel_overlap]
    return new_weights


def adapt_fc_weights(old_weights, new_shape):
    """
    Adapt old fully-connected weights (old_shape: (old_out, old_in))
    to new_shape (new_out, new_in) by copying the overlapping region.
    """
    new_weights = np.random.randn(*new_shape) * 0.1
    old_out, old_in = old_weights.shape
    new_out, new_in = new_shape
    out_overlap = min(old_out, new_out)
    in_overlap = min(old_in, new_in)
    new_weights[:out_overlap, :in_overlap] = old_weights[:out_overlap, :in_overlap]
    return new_weights


def adapt_biases(old_biases, new_length):
    """
    Adapt biases: copy as many values as possible from old_biases into a new
    bias vector of length new_length.
    """
    new_biases = np.random.randn(new_length) * 0.1
    overlap = min(old_biases.shape[0], new_length)
    new_biases[:overlap] = old_biases[:overlap]
    return new_biases


# ============================================================================
# Helper functions to crossover weights and biases.
# ============================================================================

def _crossover_matrix(m1, m2, adapt_func, target_shape):
    """
    Performs element-wise crossover of two matrices.
    If shapes differ, adapt m2 to target_shape.
    """
    # Ensure m2 has the same shape as target (assume m1 is already target_shape)
    if m2.shape != target_shape:
        m2 = adapt_func(m2, target_shape)
    mask = np.random.rand(*target_shape) < 0.5
    return np.where(mask, m1, m2)


def _crossover_vector(v1, v2, target_length):
    """
    Performs element-wise crossover of two vectors.
    If lengths differ, adapt v2.
    """
    if v2.shape[0] != target_length:
        v2 = np.random.uniform(v1.min(), v1.max(), size=(target_length,))
    mask = np.random.rand(target_length) < 0.5
    return np.where(mask, v1, v2)


# ============================================================================
# Helper functions to calculate distances between networks.
# ============================================================================

def node_cost(n1, n2):
    """
    Compute a substitution cost between two nodes (layer attributes).
    """
    if n1.get('type') != n2.get('type'):
        return 1.0  # maximum penalty for mismatched types

    cost = 0.0

    # For convolutional layers:
    if n1.get('type') == 'conv':
        cost += abs(n1.get('kernel_size', 0) - n2.get('kernel_size', 0)) / max(n1.get('kernel_size', 1),
                                                                               n2.get('kernel_size', 1))
        cost += abs(n1.get('stride', 0) - n2.get('stride', 0)) / max(n1.get('stride', 1), n2.get('stride', 1))

        # Handle padding specially to avoid division by zero
        pad1 = n1.get('padding', 0)
        pad2 = n2.get('padding', 0)
        if pad1 == 0 and pad2 == 0:
            diff_padding = 0.0
        else:
            diff_padding = abs(pad1 - pad2) / max(pad1, pad2)
        cost += diff_padding

        cost += abs(n1.get('out_channels', 0) - n2.get('out_channels', 0)) / max(n1.get('out_channels', 1),
                                                                                 n2.get('out_channels', 1))
        cost += abs(n1.get('in_channels', 0) - n2.get('in_channels', 0)) / max(n1.get('in_channels', 1),
                                                                               n2.get('in_channels', 1))
        cost += abs(n1.get('input_size', 0) - n2.get('input_size', 0)) / max(n1.get('input_size', 1),
                                                                             n2.get('input_size', 1))

    # For fully-connected layers:
    elif n1.get('type') == 'fc':
        cost += abs(n1.get('fc_layer_size', 0) - n2.get('fc_layer_size', 0)) / max(n1.get('fc_layer_size', 1),
                                                                                   n2.get('fc_layer_size', 1))
        cost += abs(n1.get('input_size', 0) - n2.get('input_size', 0)) / max(n1.get('input_size', 1),
                                                                             n2.get('input_size', 1))

    # Compare activation functions: if they differ, add a fixed penalty.
    if n1.get('activation') != n2.get('activation'):
        cost += 1.0

    return cost


def build_architecture_graph(layer_config):
    from neat_system.cnn_genome import CNNConvGene, CNNPoolGene, CNNFCGene
    """
    Build a directed graph representation from a list of layer genes.
    Each gene is converted to a node with attributes.
    """
    G = nx.DiGraph()
    for i, gene in enumerate(layer_config):
        attr = {}
        if isinstance(gene, CNNConvGene):
            attr['type'] = 'conv'
            attr['kernel_size'] = gene.kernel_size
            attr['stride'] = gene.stride
            attr['padding'] = gene.padding
            attr['out_channels'] = gene.out_channels
            attr['in_channels'] = gene.in_channels
            attr['input_size'] = gene.input_size
            attr['activation'] = gene.activation
        elif isinstance(gene, CNNPoolGene):
            attr['type'] = 'pool'
            attr['pool_size'] = gene.pool_size
            attr['stride'] = gene.stride
            attr['pool_type'] = gene.pool_type
            attr['in_channels'] = gene.in_channels
            attr['input_size'] = gene.input_size
        elif isinstance(gene, CNNFCGene):
            attr['type'] = 'fc'
            attr['fc_layer_size'] = gene.fc_layer_size
            attr['input_size'] = gene.input_size
            attr['activation'] = gene.activation
        else:
            attr['type'] = 'unknown'
        G.add_node(i, **attr)
        if i > 0:
            G.add_edge(i - 1, i)
    return G


def graph_architecture_distance(net1, net2):
    """
    Compute a graph-based architecture distance between two networks by building
    graphs from their layer_config and computing an approximate graph edit distance.
    """
    G1 = build_architecture_graph(net1.layer_config)
    G2 = build_architecture_graph(net2.layer_config)

    ged = nx.graph_edit_distance(
        G1, G2,
        node_subst_cost=lambda n1, n2: node_cost(n1, n2),
        node_del_cost=lambda u: 1.0,
        node_ins_cost=lambda u: 1.0
    )
    # In case ged is None, fall back to the difference in node count.
    if ged is None:
        ged = abs(len(G1.nodes()) - len(G2.nodes()))
    return ged


def weight_distribution_distance(weights1, weights2,
                                 alpha, beta, gamma, delta,
                                 norm_wd=1.0, norm_skew=1.0, norm_kurt=1.0, norm_quant=1.0,
                                 quantiles=[0.25, 0.5, 0.75]):
    """
    Compute a hybrid distance between two weight arrays based on:
      - Wasserstein distance
      - Absolute difference in skewness
      - Absolute difference in kurtosis
      - L2 distance between selected quantiles

    Each metric is normalized by a given factor so that they contribute on similar scales.

    Returns:
      Individual contributions and the combined weighted sum.
    """
    w1 = weights1.flatten()
    w2 = weights2.flatten()

    # Wasserstein distance
    wd = wasserstein_distance(w1, w2)
    # Skewness difference
    skew_diff = np.abs(skew(w1) - skew(w2))
    # Kurtosis difference
    kurt_diff = np.abs(kurtosis(w1) - kurtosis(w2))
    # Quantile difference: compute specified quantiles and take the L2 norm
    q1 = np.quantile(w1, quantiles)
    q2 = np.quantile(w2, quantiles)
    quant_diff = np.linalg.norm(q1 - q2)

    # Normalize each component
    wd_norm = wd / norm_wd if norm_wd else 0
    skew_norm = skew_diff / norm_skew if norm_skew else 0
    kurt_norm = kurt_diff / norm_kurt if norm_kurt else 0
    quant_norm = quant_diff / norm_quant if norm_quant else 0

    # Compute contributions
    wd_contrib = alpha * wd_norm
    skew_contrib = beta * skew_norm
    kurt_contrib = gamma * kurt_norm
    quant_contrib = delta * quant_norm

    # Compute total combined distance
    combined_distance = wd_contrib + skew_contrib + kurt_contrib + quant_contrib

    return wd_contrib, skew_contrib, kurt_contrib, quant_contrib, combined_distance


def connection_distance(net1, net2, alpha=1.5, beta=0.3, gamma=0.05, delta=1.5):
    """
    Compute a connection distance between two networks based on weight distributions.
    Prints the contribution percentages of Wasserstein, skewness, kurtosis, and quantiles.
    """
    layers1 = net1.layer_config
    layers2 = net2.layer_config
    n = min(len(layers1), len(layers2))
    total_wd, total_skew, total_kurt, total_quant, total_distance = 0, 0, 0, 0, 0
    count = 0

    for i in range(n):
        g1, g2 = layers1[i], layers2[i]

        if hasattr(g1, 'weights') and hasattr(g2, 'weights'):
            try:
                wd, skew_c, kurt_c, quant_c, combined = weight_distribution_distance(
                    g1.weights, g2.weights, alpha, beta, gamma, delta
                )
                total_wd += wd
                total_skew += skew_c
                total_kurt += kurt_c
                total_quant += quant_c
                total_distance += combined
            except Exception:
                total_distance += 1.0  # Fallback penalty
            count += 1

        if hasattr(g1, 'biases') and hasattr(g2, 'biases'):
            try:
                wd, skew_c, kurt_c, quant_c, combined = weight_distribution_distance(
                    g1.biases, g2.biases, alpha, beta, gamma, delta
                )
                total_wd += wd
                total_skew += skew_c
                total_kurt += kurt_c
                total_quant += quant_c
                total_distance += combined
            except Exception:
                total_distance += 1.0
            count += 1

    if count > 0:
        total_distance /= count
        total_wd /= count
        total_skew /= count
        total_kurt /= count
        total_quant /= count

    # Compute contribution percentages
    total_sum = total_wd + total_skew + total_kurt + total_quant
    if total_sum > 0:
        wd_pct = total_wd / total_sum * 100
        skew_pct = total_skew / total_sum * 100
        kurt_pct = total_kurt / total_sum * 100
        quant_pct = total_quant / total_sum * 100
    else:
        wd_pct = skew_pct = kurt_pct = quant_pct = 0

    # Print detailed breakdown
    # print("\n🔹 **Distance Breakdown**:")
    # print(f"  Wasserstein: {wd_pct:.2f}% ({total_wd:.4f})")
    # print(f"  Skewness:    {skew_pct:.2f}% ({total_skew:.4f})")
    # print(f"  Kurtosis:    {kurt_pct:.2f}% ({total_kurt:.4f})")
    # print(f"  Quantiles:   {quant_pct:.2f}% ({total_quant:.4f})")
    # print(f"  => **Combined: {total_distance:.4f}**\n")

    return total_distance
