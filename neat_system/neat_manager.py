import configparser
import neat

from metrics import visualize
from neat_system.cnn_genome import CNNGenome
from neat_system.weight_reporter import WeightStatsReporter
from neat_system.cnn_layers import global_innovation_registry


class NeatManager:
    """
    Manages the NEAT evolutionary process including configuration updates,
    population creation, genome/agent mapping, and evolution steps.
    """

    def __init__(self,
                 neat_config_path: str,
                 evolution_config: dict,
                 board_size: int,
                 ship_sizes: list):
        """
        Initialize the NeatManager.

        Parameters:
            neat_config_path (str): Path to the NEAT configuration file.
            evolution_config (dict): Evolution configuration parameters.
            board_size (int): Board dimensions for the game.
            ship_sizes (list): List of ship sizes to calculate fitness thresholds.
        """
        self.neat_config_path = neat_config_path
        self.evolution_config = evolution_config
        self.board_size = board_size
        self.ship_sizes = ship_sizes

        # Update NEAT configuration based on current board and evolution parameters.
        self._update_neat_config()

        # Initialize the NEAT configuration and population.
        self.config = self._initialize_neat_config()
        self.population = self._create_population()
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        self.weight_stats_reporter = WeightStatsReporter()
        self.population.add_reporter(self.weight_stats_reporter)

        # Mapping from genome id/index to (genome, agent)
        self.agents_mapping = {}

    def _update_neat_config(self) -> None:
        """
        Update the NEAT configuration file with current parameters.
        """
        cp = configparser.ConfigParser()
        cp.read(self.neat_config_path)

        # Set fitness threshold and population size for the NEAT process.
        cp["NEAT"]["fitness_threshold"] = str(0.8*(self.board_size ** 2 - sum(self.ship_sizes)))
        cp["NEAT"]["pop_size"] = str(self.evolution_config["search_population"]["size"])

        # Update CNNGenome parameters.
        cp["CNNGenome"]["input_size"] = str(self.board_size)
        cp["CNNGenome"]["output_size"] = str(self.board_size ** 2)

        # Write the updated configuration back to disk.
        with open(self.neat_config_path, "w") as f:
            cp.write(f)

    def _initialize_neat_config(self) -> neat.Config:
        """
        Load and return the NEAT configuration object using the updated config file.
        """
        return neat.Config(
            CNNGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.neat_config_path
        )

    def _create_population(self) -> neat.Population:
        """
        Create and return a NEAT population.
        """
        pop = neat.Population(self.config)
        # Optionally, add the standard reporter to see output in stdout.
        pop.add_reporter(neat.StdOutReporter(True))
        return pop

    def evolve_one_generation(self) -> bool:
        """
        Performs one generation of evolution in the NEAT population.

        Returns:
            bool: True if a solution meeting the fitness threshold is found, else False.
        """
        # Find the best genome in the current population.
        best = max(self.population.population.values(), key=lambda g: g.fitness)
        self.population.reporters.post_evaluate(self.config, self.population.population, self.population.species, best)

        # Update the best genome ever seen.
        if self.population.best_genome is None or best.fitness > self.population.best_genome.fitness:
            self.population.best_genome = best

        # Check for termination by fitness threshold.
        if not self.config.no_fitness_termination:
            best_fitness = self.config.fitness_criterion(g.fitness for g in self.population.population.values())
            if best_fitness >= self.config.fitness_threshold:
                self.population.reporters.found_solution(self.config, self.population.generation, best)
                return True  # A solution was found!

        # Reproduce the next generation.
        self.population.population = self.population.reproduction.reproduce(self.config,
                                                                            self.population.species,
                                                                            self.config.pop_size,
                                                                            self.population.generation)
        # Handle the case of complete extinction.
        if not self.population.species.species:
            self.population.reporters.complete_extinction()
            if self.config.reset_on_extinction:
                self.population.population = self.population.reproduction.create_new(
                    self.config.genome_type,
                    self.config.genome_config,
                    self.config.pop_size
                )
            else:
                raise neat.CompleteExtinctionException()

        # Speciate the newly created population.

        # full NEAT speciation
        self.population.species.speciate(
                self.config,
                self.population.population,
                self.population.generation)

        self.population.reporters.end_generation(self.config, self.population.population, self.population.species)

        # Increment the generation counter.
        self.population.generation += 1
        return False

    def visualize_results(self):
        """Generate visualizations for NEAT results."""
        visualize.visualize_hof(statistics=self.stats)
        visualize.plot_weight_stats(self.weight_stats_reporter.get_weight_stats())
        visualize.plot_species_weight_stats(
            self.weight_stats_reporter.get_species_weight_stats()
        )
        species_analysis = visualize.analyze_species_from_population(self.population.species)
        visualize.plot_species_analysis(species_analysis)
        visualize.visualize_species(self.stats)
        visualize.plot_stats(
            statistics=self.stats,
            best_possible=0.8*(self.board_size ** 2 - sum(self.ship_sizes)),
            ylog=False,
            view=True,
        )
        visualize.plot_fitness_boxplot(self.stats)

        visualize.plot_innovation_registry(global_innovation_registry)

        best_genomes = self.stats.best_genomes(5)
        genomes = [(genome, "") for genome in best_genomes]
        visualize.plot_multiple_genomes(genomes, "Best Genomes")
        visualize.plot_multiple_genomes_active(genomes, "Best Genomes Active")

