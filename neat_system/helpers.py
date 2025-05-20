import copy
from random import random

import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis


# ============================================================================
# Helper functions to crossover between genes
# ============================================================================
def _crossover_by_key(config, genes1, genes2, fitness1, fitness2):

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
    Computes the NEAT-style distance for one gene type between two genomes.
    - self_genes / other_genes: dicts mapping innovation key → gene
    - gene_distance_func(g1, g2, config) must return (struct_diff, weight_diff)
    - Excess genes: those with key > max innovation of the other genome
    - Disjoint genes: non-matching keys within the overlapping innovation range
    """
    # If neither genome has this gene type, distance is zero.
    if not self_genes and not other_genes:
        return 0.0

    # Gather all innovation keys
    all_keys = set(self_genes) | set(other_genes)
    max_self  = max(self_genes, default=0)
    max_other = max(other_genes, default=0)

    # Counters
    excess = 0
    disjoint = 0
    matching_weight_sum = 0.0
    matching_count = 0

    for key in all_keys:
        g1 = self_genes.get(key)
        g2 = other_genes.get(key)

        if g1 is not None and g2 is not None:
            # Matching gene: unpack and accumulate only the weight diff
            _, w_diff = gene_distance_func(g1, g2, config)
            matching_weight_sum += w_diff
            matching_count += 1
        else:
            # Non-matching: decide excess vs disjoint
            if (g1 is None and key > max_self) \
                    or (g2 is None and key > max_other):
                excess += 1
            else:
                disjoint += 1

    # Average weight difference over matching genes
    avg_weight_diff = (matching_weight_sum / matching_count) if matching_count > 0 else 0.0

    # Normalization factor N (use 1 for small genomes)
    N = max(len(self_genes), len(other_genes))
    if N < getattr(config, "compatibility_threshold_N", 20):
        N = 1

    # NEAT compatibility distance
    return (excess_coeff * excess) / N \
        + (disjoint_coeff * disjoint) / N \
        + (weight_coeff * avg_weight_diff)


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

    Each metric is normalized by the provided factor.  If either array is
    constant, we skip skew/kurtosis (set them to zero) to avoid NaNs.
    """
    w1 = weights1.flatten()
    w2 = weights2.flatten()

    # Check for constant arrays
    const1 = np.allclose(w1, w1[0])
    const2 = np.allclose(w2, w2[0])

    # 1) Wasserstein
    wd = wasserstein_distance(w1, w2)

    # 2) Quantiles
    q1 = np.quantile(w1, quantiles)
    q2 = np.quantile(w2, quantiles)
    quant_diff = np.linalg.norm(q1 - q2)

    if const1 or const2:
        # No skew/kurt for constant distributions
        skew_diff = 0.0
        kurt_diff = 0.0
    else:
        # Safely compute skew/kurtosis, omitting NaNs
        skew1 = skew(w1, nan_policy='omit')
        skew2 = skew(w2, nan_policy='omit')
        skew_diff = abs(skew1 - skew2)

        kurt1 = kurtosis(w1, nan_policy='omit', fisher=False)
        kurt2 = kurtosis(w2, nan_policy='omit', fisher=False)
        kurt_diff = abs(kurt1 - kurt2)

    # Normalize components
    wd_norm    = wd    / norm_wd    if norm_wd    else 0.0
    skew_norm  = skew_diff / norm_skew  if norm_skew  else 0.0
    kurt_norm  = kurt_diff / norm_kurt  if norm_kurt  else 0.0
    quant_norm = quant_diff / norm_quant if norm_quant else 0.0

    # Weighted contributions
    wd_contrib    = alpha * wd_norm
    skew_contrib  = beta  * skew_norm
    kurt_contrib  = gamma * kurt_norm
    quant_contrib = delta * quant_norm

    combined_distance = wd_contrib + skew_contrib + kurt_contrib + quant_contrib

    return wd_contrib, skew_contrib, kurt_contrib, quant_contrib, combined_distance


def connection_distance(g1, g2, alpha=1.5, beta=0.3, gamma=0.05, delta=1.5):
    """
    Compute a connection distance between two genes’ weights and biases,
    summing the hybrid distance on each.
    """
    # Weights
    wd_w, sk_w, ku_w, qu_w, c_w = weight_distribution_distance(
        g1.weights, g2.weights, alpha, beta, gamma, delta
    )

    # Biases
    wd_b, sk_b, ku_b, qu_b, c_b = weight_distribution_distance(
        g1.biases, g2.biases, alpha, beta, gamma, delta
    )

    total_distance = c_w + c_b

    return total_distance
