import numpy as np
from neat.reporting import BaseReporter


class WeightStatsReporter(BaseReporter):
    """
    Custom reporter to track weight statistics across generations.
    In addition to global stats from the best genome, it records per-species
    weight statistics over generations.
    """

    def __init__(self):
        super().__init__()
        self.weight_stats = {}  # {generation: {mean, std, min, max}} from best genome
        self.species_weight_stats = {}  # {generation: {species_id: {mean, std, min, max}}}

    def post_evaluate(self, config, population, species, best_genome):
        # --- Global stats from the best genome ---
        stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        for gene in best_genome.layer_config:
            if hasattr(gene, 'weights'):
                w = gene.weights.flatten()
                stats['mean'].append(np.mean(w))
                stats['std'].append(np.std(w))
                stats['min'].append(np.min(w))
                stats['max'].append(np.max(w))
            if hasattr(gene, 'biases'):
                b = gene.biases.flatten()
                stats['mean'].append(np.mean(b))
                stats['std'].append(np.std(b))
                stats['min'].append(np.min(b))
                stats['max'].append(np.max(b))
        avg_stats = {key: np.mean(val) for key, val in stats.items()}
        generation = len(self.weight_stats)
        self.weight_stats[generation] = avg_stats

        # --- Species-specific stats ---
        species_stats_for_gen = {}
        # species.species is a dict: {species_id: species_object}
        for species_id, species_obj in species.species.items():
            species_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
            # species_obj.members is a dict of genomes in this species.
            for genome in species_obj.members.values():
                genome_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
                for gene in genome.layer_config:
                    if hasattr(gene, 'weights'):
                        w = gene.weights.flatten()
                        genome_stats['mean'].append(np.mean(w))
                        genome_stats['std'].append(np.std(w))
                        genome_stats['min'].append(np.min(w))
                        genome_stats['max'].append(np.max(w))
                    if hasattr(gene, 'biases'):
                        b = gene.biases.flatten()
                        genome_stats['mean'].append(np.mean(b))
                        genome_stats['std'].append(np.std(b))
                        genome_stats['min'].append(np.min(b))
                        genome_stats['max'].append(np.max(b))
                # Aggregate over all genes in the genome
                if genome_stats['mean']:
                    aggregated_genome_stats = {k: np.mean(genome_stats[k]) for k in genome_stats}
                    species_stats['mean'].append(aggregated_genome_stats['mean'])
                    species_stats['std'].append(aggregated_genome_stats['std'])
                    species_stats['min'].append(aggregated_genome_stats['min'])
                    species_stats['max'].append(aggregated_genome_stats['max'])
            if species_stats['mean']:
                aggregated_species_stats = {k: np.mean(species_stats[k]) for k in species_stats}
            else:
                aggregated_species_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            species_stats_for_gen[species_id] = aggregated_species_stats

        self.species_weight_stats[generation] = species_stats_for_gen

    def get_weight_stats(self):
        """Returns global weight statistics from best genomes."""
        return self.weight_stats

    def get_species_weight_stats(self):
        """Returns weight statistics per species for each generation."""
        return self.species_weight_stats
