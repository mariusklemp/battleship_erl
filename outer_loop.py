import json
import os
import time

from matplotlib import pyplot as plt
from neat import CompleteExtinctionException
from tqdm import tqdm
import neat
import configparser

from RBUF import RBUF
from ai.mcts import MCTS
from deap_system.placement_ga import PlacementGeneticAlgorithm
from metrics.evaluator import Evaluator
from game_logic.search_agent import SearchAgent
from game_logic.placement_agent import PlacementAgent
from game_logic.game_manager import GameManager
from ai.model import ANET
from inner_loop import InnerLoopManager
from neat_system.neat_manager import NEAT_Manager
from neat_system.cnn_genome import CNNGenome
import visualize
from neat_system.weight_reporter import WeightStatsReporter
import gc


class OuterLoopManager:
    """
    Manages the outer evolutionary loop for training Battleship agents.
    Handles both placing and search agents, with support for NEAT and standard neural networks.
    """

    def __init__(self, mcts_config_path="config/mcts_config.json",
                 evolution_config_path="config/evolution_config.json",
                 layer_config_path="ai/config.json"):
        # Load configurations
        self.mcts_config = self._load_config(mcts_config_path)
        self.evolution_config = self._load_config(evolution_config_path)
        self.layer_config = self._load_config(layer_config_path)

        # Extract common parameters
        self.board_size = self.mcts_config["board_size"]
        self.ship_sizes = self.mcts_config["ship_sizes"]
        self.run_inner_loop = True
        self.run_evolution = True

        # Initialize components
        self.game_manager = GameManager(size=self.board_size)

        self._initialize_evaluator()

        # Initialize agents and metrics tracker
        self.search_agents = []

        # Create directory for saving models
        os.makedirs("models", exist_ok=True)

    def _load_config(self, config_path):
        """Load MCTS configuration."""
        with open(config_path, "r") as f:
            return json.load(f)

    def _initialize_evaluator(self):
        """Initialize the evaluator for agent performance assessment."""
        self.evaluator = Evaluator(
            game_manager=self.game_manager,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.evolution_config.get("num_evaluation_games", 10),
        )

    def _create_nn_agent(self, model_number):
        """Create a neural network-based search agent."""

        # Create a new network instance for each model
        net = ANET(board_size=self.board_size,
                   activation="relu",
                   device="cpu",
                   layer_config=self.layer_config)

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            name=f"nn_{model_number}",
            lr=0.001,
        )
        return search_agent

    def _initialize_nn_search_agents(self):
        """Initialize search agents with neural networks."""
        population_size = self.evolution_config["search_population"]["size"]
        search_agents = []

        for i in range(population_size + 1):
            agent = self._create_nn_agent(i * 100)
            search_agents.append(agent)

        return search_agents

    def _initialize_neat(self):
        """Initialize the NEAT system for evolving neural networks."""
        # Set up NEAT configuration
        neat_config_path = os.path.join("neat_system", "config.txt")

        # Update the config file with current parameters
        self._update_neat_config(neat_config_path)

        # Load the NEAT configuration
        config = neat.Config(
            CNNGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
        )

        # Create a NEAT population
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Attach custom WeightStatsReporter
        weight_stats_reporter = WeightStatsReporter()
        p.add_reporter(weight_stats_reporter)

        # Initialize NEAT_Manager
        range_evaluations = 5  # Number of games to evaluate each genome
        neat_manager = NEAT_Manager(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            strategy_placement="random",
            strategy_search="nn_search",
            range_evaluations=range_evaluations,
            config=config,
            game_manager=self.game_manager,
        )

        self.neat_config = config
        self.neat_population = p
        self.neat_stats = stats
        self.weight_stats_reporter = weight_stats_reporter
        self.neat_manager = neat_manager

    def _update_neat_config(self, config_path):
        """Update NEAT configuration file with current parameters."""
        cp = configparser.ConfigParser()
        cp.read(config_path)

        # Update NEAT configuration
        cp["NEAT"]["fitness_threshold"] = str(self.board_size ** 2 - sum(self.ship_sizes))
        cp["NEAT"]["pop_size"] = str(self.evolution_config["search_population"]["size"])

        # Update CNNGenome configuration
        cp["CNNGenome"]["input_size"] = str(self.board_size)
        cp["CNNGenome"]["output_size"] = str(self.board_size ** 2)

        # Write updated config back to disk
        with open(config_path, "w") as f:
            cp.write(f)

    def create_search_agent_genomes(self, population, config):
        """Create a mapping from genome IDs to a tuple of (genome, search agent)."""
        mapping = {}
        for genome_id, genome in population.population.items():
            try:
                net = ANET(genome=genome, config=config)
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy="nn_search",
                    net=net,
                    optimizer="adam",
                    name=f"neat_{genome_id}_gen_0",
                    lr=0.001,
                )
                mapping[genome_id] = (genome, agent)
            except Exception as e:
                print(f"Error creating agent for genome {genome_id}: {e}")
        return mapping

    def evaluate_agents_baselines(self, search_agents, placement_agents=None, generation=0):
        """Evaluate both search and placement agents against baseline opponents."""
        print(f"\nEvaluating agents against baseline opponents...")
        self.evaluator.evaluate_search_agents(search_agents, generation)
        if placement_agents:
            self.evaluator.evaluate_placement_agents(placement_agents, generation)

    def _visualize_neat_results(self, stats, weight_stats_reporter, population):
        """Generate visualizations for NEAT results."""
        print("\nPlotting NEAT statistics...")
        visualize.visualize_hof(statistics=stats)
        visualize.plot_weight_stats(weight_stats_reporter.get_weight_stats())
        visualize.plot_species_weight_stats(
            weight_stats_reporter.get_species_weight_stats()
        )
        species_analysis = visualize.analyze_species_from_population(population.species)
        visualize.plot_species_analysis(species_analysis)
        visualize.visualize_species(stats)
        visualize.plot_stats(
            statistics=stats,
            best_possible=(self.board_size ** 2 - sum(self.ship_sizes)),
            ylog=False,
            view=True,
        )
        visualize.plot_fitness_boxplot(stats)

        from neat_system.cnn_layers import global_innovation_registry
        visualize.plot_innovation_registry(global_innovation_registry)

        best_genomes = stats.best_genomes(5)
        genomes = [(genome, "") for genome in best_genomes]
        visualize.plot_multiple_genomes(genomes, "Best Genomes")

    def run(self):
        """Run the outer evolutionary loop and record timings and instance counts."""
        num_generations = self.evolution_config["evolution"]["num_generations"]
        rbuf = RBUF(max_len=self.mcts_config["replay_buffer"]["max_size"])
        if self.mcts_config["replay_buffer"]["load_from_file"]:
            rbuf.init_from_file(file_path=self.mcts_config["replay_buffer"]["file_path"])

        if self.run_evolution:
            self._initialize_neat()
            print(f"Initialized neat population (genomes) search agents")
            placement_ga = PlacementGeneticAlgorithm(
                game_manager=self.game_manager,
                board_size=self.board_size,
                ship_sizes=self.ship_sizes,
                population_size=self.evolution_config["placing_population"]["size"],
                num_generations=self.evolution_config["evolution"]["num_generations"],
                elite_size=self.evolution_config["placing_population"]["elite_size"],
                MUTPB=self.evolution_config["placing_population"]["mutation_probability"],
                TOURNAMENT_SIZE=self.evolution_config["placing_population"]["tournament_size"],
            )
            placement_ga.initialize_placing_population(self.evolution_config["placing_population"]["size"])
            print(f"\nInitialized {len(placement_ga.pop_placing_agents)} placement agents...")
        else:
            placement_ga = None

        print("\nStarting evolution...")

        # Define the classes you wish to track
        tracked_classes = (
            GameManager,
            RBUF,
            Evaluator,
            MCTS,
            neat.Population,
            PlacementGeneticAlgorithm,
            PlacementAgent,
            SearchAgent,
            ANET,
            CNNGenome,
        )

        # Data structures for timing and instance counts.
        timings = {
            'baseline_evaluation': [],
            'inner_loop_training': [],
            'placement_evolution': [],
            'neat_evolution': [],
        }
        total_objects_list = []
        instance_counts = {cls.__name__: [] for cls in tracked_classes}

        for gen in range(num_generations):
            gen_start_time = time.perf_counter()
            print(f"\n=== Generation {gen + 1}/{num_generations} ===")

            # --- Step 1: Garbage Collection and Instance Counting ---
            gc.collect()
            objects = gc.get_objects()
            total_objs = len(objects)
            print("Total objects:", total_objs)
            total_objects_list.append(total_objs)

            # Count instances of tracked classes
            count = {}
            for obj in objects:
                if isinstance(obj, tracked_classes):
                    obj_type = type(obj)
                    count[obj_type] = count.get(obj_type, 0) + 1

            # Record counts for each class (even if zero)
            for cls in tracked_classes:
                cls_name = cls.__name__
                instance_counts[cls_name].append(count.get(cls, 0))

            # Optionally, print counts for this generation
            for cls, cnt in count.items():
                cls_name = getattr(cls, '__name__', str(cls))
                print(f"{cls_name}: {cnt}")

            # --- Step 2: Create Search Agents (NEAT or NN) ---
            if self.run_evolution:
                self.neat_population.reporters.start_generation(self.neat_population.generation)
                search_agents_mapping = self.create_search_agent_genomes(self.neat_population, self.neat_config)
                self.search_agents = [agent for (genome, agent) in search_agents_mapping.values()]
            else:
                self.search_agents = self._initialize_nn_search_agents()

            # --- Step 3: Baseline Evaluation ---
            step_start = time.perf_counter()
            if self.run_evolution:
                self.evaluate_agents_baselines(self.search_agents, placement_ga.pop_placing_agents, gen)
            else:
                self.evaluate_agents_baselines(self.search_agents, gen)
            step_end = time.perf_counter()
            timings['baseline_evaluation'].append(step_end - step_start)

            # --- Step 4: Inner Loop Training ---
            step_start = time.perf_counter()
            if self.run_inner_loop:
                print(f"\nInner loop: Training {len(self.search_agents)} search agents")
                inner_loop_manager = InnerLoopManager(game_manager=self.game_manager)
                for i, search_agent in tqdm(enumerate(self.search_agents), desc="Training search agents",
                                            total=len(self.search_agents)):
                    print(f"\nTraining search agent {i + 1}/{len(self.search_agents)}")
                    inner_loop_manager.run(search_agent, rbuf=rbuf)
            step_end = time.perf_counter()
            timings['inner_loop_training'].append(step_end - step_start)

            # --- Step 5: Placement Evaluation and Evolution ---
            step_start = time.perf_counter()
            if self.run_evolution:
                print("Evolving placing agents...")
                placement_ga.trigger_evaluate_population(self.search_agents)
                placement_ga.evolve()
                step_end = time.perf_counter()
                timings['placement_evolution'].append(step_end - step_start)

                print("Evolving search agents...")
                step_start = time.perf_counter()
                for genome_id, (genome, search_agent) in search_agents_mapping.items():
                    total_fitness = 0.0
                    for placement_agent in placement_ga.pop_placing_agents:
                        fitness = self.neat_manager.evaluate(self.game_manager, search_agent, placement_agent)
                        total_fitness += fitness
                    genome.fitness = total_fitness / len(placement_ga.pop_placing_agents)

                self.evolve_neat()
                step_end = time.perf_counter()
                timings['neat_evolution'].append(step_end - step_start)

            gen_end_time = time.perf_counter()
            print(f"Generation {gen + 1} took {gen_end_time - gen_start_time:.2f} seconds.")

        # After all generations, plot the timings and instance counts.
        self._plot_timings_and_instance_counts(timings, total_objects_list, instance_counts)

        # Plot any additional metrics you already have.
        self._generate_visualizations(placement_ga)

    def _plot_timings_and_instance_counts(self, timings, total_objects_list, instance_counts):
        """Plot the timing data and instance counts per generation."""
        generations = range(1, len(total_objects_list) + 1)

        # Plot timings for each step
        plt.figure(figsize=(10, 6))
        for key, time_list in timings.items():
            plt.plot(generations, time_list, label=key)
        plt.xlabel("Generation")
        plt.ylabel("Time (seconds)")
        plt.title("Timing per Generation")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot total object counts per generation
        plt.figure(figsize=(10, 6))
        plt.plot(generations, total_objects_list, label="Total Objects", marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Number of Objects")
        plt.title("Total Object Count per Generation")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot instance counts for each tracked class
        plt.figure(figsize=(10, 6))
        for cls_name, counts in instance_counts.items():
            plt.plot(generations, counts, label=cls_name)
        plt.xlabel("Generation")
        plt.ylabel("Instance Count")
        plt.title("Instance Counts per Generation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def evolve_neat(self):
        # Shortcut reference for neat population
        p = self.neat_population

        # Get the best genome from the current population.
        best = max(p.population.values(), key=lambda g: g.fitness)
        p.reporters.post_evaluate(p.config, p.population, p.species, best)

        # Update the best genome ever seen.
        if p.best_genome is None or best.fitness > p.best_genome.fitness:
            p.best_genome = best

        # Check for fitness termination condition.
        if not p.config.no_fitness_termination:
            fv = p.fitness_criterion(g.fitness for g in p.population.values())
            if fv >= p.config.fitness_threshold:
                p.reporters.found_solution(p.config, p.generation, best)
                return True  # Signal that a solution was found.

        # Create the next generation.
        p.population = p.reproduction.reproduce(p.config, p.species, p.config.pop_size, p.generation)

        # Check for complete extinction.
        if not p.species.species:
            p.reporters.complete_extinction()
            if p.config.reset_on_extinction:
                p.population = p.reproduction.create_new(p.config.genome_type,
                                                         p.config.genome_config,
                                                         p.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Partition the new population into species.
        p.species.speciate(p.config, p.population, p.generation)
        p.reporters.end_generation(p.config, p.population, p.species)

        # Increment generation counter.
        p.generation += 1

        return False  # Signal that evolution should continue.

    def _generate_visualizations(self, placement_ga):
        """Generate all visualizations at the end of training."""
        # Plot general metrics
        placement_ga.plot_metrics()
        self.evaluator.plot_metrics_search()

        # Plot NEAT-specific visualizations if NEAT was used
        if self.run_evolution:
            self.evaluator.plot_metrics_placement()
            self._visualize_neat_results(self.neat_stats, self.weight_stats_reporter, self.neat_population)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and run the outer loop manager
    outer_loop = OuterLoopManager()
    outer_loop.run()
