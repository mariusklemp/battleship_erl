import copy
from random import random

import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis


# ============================================================================
# Helper functions to crossover between genes
# ============================================================================
def _crossover_by_key(config, genes1, genes2, fitness1, fitness2):
    """
    Given two lists of genes (sorted by key), performs NEAT-style crossover:
      - For matching genes (same key): combine weights and biases from both parents.
        If either gene is disabled, the child gene is disabled with 75% probability.
      - For disjoint/excess genes: inherit from the fitter parent.
    Returns a new list of child genes (copies).
    """
    child_genes = []

    # Sort parents' genes by key.
    g1 = sorted(genes1, key=lambda x: x.key)
    g2 = sorted(genes2, key=lambda x: x.key)

    i, j = 0, 0
    while i < len(g1) and j < len(g2):
        if g1[i].key == g2[j].key:
            gene1, gene2 = g1[i], g2[j]
            chosen_gene = copy.deepcopy(gene1)
            # Use helper to combine weights and biases.
            if config.crossover_weights:
                new_weights, new_biases = crossover_gene_parameters(gene1, gene2)
                if new_weights is not None:
                    chosen_gene.weights = new_weights
                if new_biases is not None:
                    chosen_gene.biases = new_biases

            # If either parent's gene is disabled, disable the child's gene with 75% probability.
            if (not gene1.enabled or not gene2.enabled) and random() < 0.75:
                chosen_gene.enabled = False
            else:
                chosen_gene.enabled = True

            child_genes.append(chosen_gene)
            i += 1
            j += 1
        elif g1[i].key < g2[j].key:
            # Disjoint gene from g1.
            if fitness1 >= fitness2:
                child_genes.append(copy.deepcopy(g1[i]))
            i += 1
        else:
            # Disjoint gene from g2.
            if fitness2 >= fitness1:
                child_genes.append(copy.deepcopy(g2[j]))
            j += 1

    # Inherit excess genes from the fitter parent.
    while i < len(g1):
        if fitness1 >= fitness2:
            child_genes.append(copy.deepcopy(g1[i]))
        i += 1

    while j < len(g2):
        if fitness2 >= fitness1:
            child_genes.append(copy.deepcopy(g2[j]))
        j += 1

    return child_genes


def crossover_gene_parameters(gene1, gene2):
    """
    Combine the weights and biases of two matching genes.
    Returns new_weights and new_biases.

    If the shapes differ, adapt gene2's parameters to gene1's shape before combining.
    """
    new_weights = None
    new_biases = None

    # Handle weights:
    if hasattr(gene1, 'weights') and hasattr(gene2, 'weights'):
        if gene1.weights.shape == gene2.weights.shape:
            new_weights = (gene1.weights + gene2.weights) / 2.0
        else:
            # If shapes differ, adapt gene2's weights to gene1's shape.
            if gene1.__class__.__name__ == "CNNConvGene":
                adapted = adapt_conv_weights(gene2.weights, gene1.weights.shape)
            elif gene1.__class__.__name__ == "CNNFCGene":
                adapted = adapt_fc_weights(gene2.weights, gene1.weights.shape)
            else:
                # Fallback: if no adapter is available, use gene1's weights shape.
                adapted = gene2.weights
            new_weights = (gene1.weights + adapted) / 2.0

    # Handle biases:
    if hasattr(gene1, 'biases') and hasattr(gene2, 'biases'):
        if gene1.biases.shape == gene2.biases.shape:
            new_biases = (gene1.biases + gene2.biases) / 2.0
        else:
            adapted = adapt_biases(gene2.biases, gene1.biases.shape[0])
            new_biases = (gene1.biases + adapted) / 2.0

    return new_weights, new_biases


# ============================================================================
# Helper functions to compute distance between genes
# ============================================================================
def compute_gene_type_distance(self_genes, other_genes, gene_distance_func, config,
                               excess_coeff, disjoint_coeff, weight_coeff):
    """
    Computes the distance for a given gene type between two genomes,
    distinguishing excess and disjoint genes.
    """
    # If both sets are empty, distance is zero.
    if not self_genes and not other_genes:
        return 0.0

    # Maximum innovation numbers in each genome.
    max_self_innov = max(self_genes.keys()) if self_genes else 0
    max_other_innov = max(other_genes.keys()) if other_genes else 0

    matching_distance_sum = 0.0
    matching_count = 0
    excess = 0
    disjoint = 0

    # Consider all gene keys present in either genome.
    all_keys = set(self_genes.keys()).union(other_genes.keys())
    for key in all_keys:
        gene_self = self_genes.get(key)
        gene_other = other_genes.get(key)
        if gene_self is not None and gene_other is not None:
            # Matching (homologous) genes.
            matching_distance_sum += gene_distance_func(gene_self, gene_other, config)
            matching_count += 1
        else:
            # Gene present in only one genome.
            if gene_self is None:
                # Gene missing in self.
                if key > max_self_innov:
                    excess += 1
                else:
                    disjoint += 1
            else:
                # Gene missing in other.
                if key > max_other_innov:
                    excess += 1
                else:
                    disjoint += 1

    avg_weight_diff = matching_distance_sum / matching_count if matching_count > 0 else 0.0
    # Normalize by the number of genes (use 1 if genomes are very small).
    N = max(len(self_genes), len(other_genes))
    N = N if N >= 20 else 1

    return (excess_coeff * excess) / N + (disjoint_coeff * disjoint) / N + (weight_coeff * avg_weight_diff)


# ============================================================================
# Helper functions to mutate weights
# ============================================================================

def _mutate_weights_conv(conv_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                         weight_min_value, weight_max_value, config):
    # Mutate weights and biases using the helper.
    weights = _mutate_array(conv_gene.weights, weight_mutate_rate, weight_mutate_power,
                            weight_replace_rate, weight_min_value, weight_max_value, config)
    biases = _mutate_array(conv_gene.biases, weight_mutate_rate, weight_mutate_power,
                           weight_replace_rate, weight_min_value, weight_max_value, config)

    return weights, biases


def _mutate_weights_fc(fc_gene, weight_mutate_rate, weight_mutate_power, weight_replace_rate,
                       weight_min_value, weight_max_value, config):
    # Mutate weights and biases using the same helper.
    weights = _mutate_array(fc_gene.weights, weight_mutate_rate, weight_mutate_power,
                            weight_replace_rate, weight_min_value, weight_max_value, config)
    biases = _mutate_array(fc_gene.biases, weight_mutate_rate, weight_mutate_power,
                           weight_replace_rate, weight_min_value, weight_max_value, config)

    return weights, biases


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
# Helper functions to calculate distances between networks.
# ============================================================================

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


def connection_distance(g1, g2, alpha=1.5, beta=0.3, gamma=0.05, delta=1.5):
    """
    Compute a connection distance between two networks based on weight distributions.
    Prints the contribution percentages of Wasserstein, skewness, kurtosis, and quantiles.
    """

    total_wd, total_skew, total_kurt, total_quant, total_distance = 0, 0, 0, 0, 0

    wd, skew_c, kurt_c, quant_c, combined = weight_distribution_distance(
        g1.weights, g2.weights, alpha, beta, gamma, delta
    )
    total_wd += wd
    total_skew += skew_c
    total_kurt += kurt_c
    total_quant += quant_c
    total_distance += combined

    wd, skew_c, kurt_c, quant_c, combined = weight_distribution_distance(
        g1.biases, g2.biases, alpha, beta, gamma, delta
    )
    total_wd += wd
    total_skew += skew_c
    total_kurt += kurt_c
    total_quant += quant_c
    total_distance += combined

    return total_distance
