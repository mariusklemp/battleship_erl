import json
import os
from tqdm import tqdm
import neat
import configparser

from RBUF import RBUF
from deap_system.evolution import Evolution
from metrics.evolution_evaluator import EvolutionEvaluator
from game_logic.search_agent import SearchAgent
from game_logic.game_manager import GameManager
from ai.model import ANET
from deap_system.search_metrics import SearchMetricsTracker
from run_mcts import run_mcts_inner_loop, load_config
from neat_system.neat_manager import NEAT_Manager
from neat_system.cnn_genome import CNNGenome
import visualize
from neat_system.weight_reporter import WeightStatsReporter


class OuterLoopManager:
    """
    Manages the outer evolutionary loop for training Battleship agents.
    Handles both placing and search agents, with support for NEAT and standard neural networks.
    """
    
    def __init__(self, mcts_config_path="config/mcts_config.json", 
                 evolution_config_path="config/evolution_config.json",
                 layer_config_path="ai/config.json"):
        # Load configurations
        self.mcts_config = self._load_mcts_config(mcts_config_path)
        self.evolution_config = self._load_evolution_config(evolution_config_path)
        self.layer_config = self._load_layer_config(layer_config_path)
        
        # Extract common parameters
        self.board_size = self.mcts_config["board_size"]
        self.ship_sizes = self.mcts_config["ship_sizes"]
        self.run_mcts = True
        self.use_neat = True
        
        # Initialize components
        self.game_manager = GameManager(size=self.board_size)
        self._initialize_environment()
        self._initialize_evaluator()
        
        # Initialize agents and metrics tracker
        self.search_agents = []
        self.search_metrics = None
        
        # Create directory for saving models
        os.makedirs("models", exist_ok=True)
    
    def _load_mcts_config(self, config_path):
        """Load MCTS configuration."""
        return load_config(config_path)
    
    def _load_evolution_config(self, config_path):
        """Load evolution configuration from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_layer_config(self, config_path):
        """Load neural network layer configuration from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _initialize_environment(self):
        """Initialize the evolution environment for placing agents."""
        self.environment = Evolution(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            population_size=self.evolution_config["placing_population"]["size"],
            num_generations=self.evolution_config["evolution"]["num_generations"],
            elite_size=self.evolution_config["placing_population"]["elite_size"],
            MUTPB=self.evolution_config["placing_population"]["mutation_probability"],
            TOURNAMENT_SIZE=self.evolution_config["placing_population"]["tournament_size"],
        )
    
    def _initialize_evaluator(self):
        """Initialize the evaluator for agent performance assessment."""
        self.evaluator = EvolutionEvaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.evolution_config.get("num_evaluation_games", 10),
        )
    
    def _create_nn_agent(self, index, model_number):
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
            agent = self._create_nn_agent(i, i * 100)
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
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Attach custom WeightStatsReporter
        weight_stats_reporter = WeightStatsReporter()
        population.add_reporter(weight_stats_reporter)
        
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
        
        return config, population, stats, weight_stats_reporter, neat_manager
    
    def _update_neat_config(self, config_path):
        """Update NEAT configuration file with current parameters."""
        cp = configparser.ConfigParser()
        cp.read(config_path)
        
        # Update NEAT configuration
        cp["NEAT"]["fitness_threshold"] = str(self.board_size**2 - sum(self.ship_sizes))
        cp["NEAT"]["pop_size"] = str(self.evolution_config["search_population"]["size"])
        
        # Update CNNGenome configuration
        cp["CNNGenome"]["input_size"] = str(self.board_size)
        cp["CNNGenome"]["output_size"] = str(self.board_size**2)
        
        # Write updated config back to disk
        with open(config_path, "w") as f:
            cp.write(f)
    
    def _initialize_neat_search_agents(self, population, config):
        """Create search agents from NEAT genomes."""
        search_agents = []
        
        for i, (genome_id, genome) in enumerate(population.population.items()):
            try:
                net = ANET(genome=genome, config=config)
                agent = SearchAgent(
                    board_size=self.board_size,
                    strategy="nn_search",
                    net=net,
                    optimizer="adam",
                    name=f"neat_{i}_gen_0",
                    lr=0.001,
                )
                search_agents.append(agent)
            except Exception as e:
                print(f"Error creating agent for genome {genome_id}: {e}")
                
        return search_agents
    
    def _train_search_agents_with_mcts(self, rbuf):
        """Train search agents using Monte Carlo Tree Search."""
        print("\nTraining search agents against placing population...")
        for i, search_agent in tqdm(
            enumerate(self.search_agents),
            desc="Training search agents",
            total=len(self.search_agents),
        ):
            print(f"\nTraining search agent {i + 1}/{len(self.search_agents)}")
            for _ in range(self.evolution_config["evolution"]["inner_loop_epochs"]):
                run_mcts_inner_loop(
                    self.environment.game_manager,
                    search_agent,
                    simulations_number=self.mcts_config["mcts"]["simulations_number"],
                    exploration_constant=self.mcts_config["mcts"]["exploration_constant"],
                    batch_size=self.mcts_config["training"]["batch_size"],
                    device=self.mcts_config["device"],
                    sizes=self.ship_sizes,
                    placement_agents=self.environment.pop_placing_agents,
                    epochs=self.mcts_config["training"]["epochs"],
                    rbuf=rbuf,
                )
    
    def _evaluate_neat_genomes(self, neat_manager, population, config, generation):
        """Evaluate NEAT genomes and save the best one."""
        print("Evaluating NEAT genomes...")
        neat_manager.set_placement_agents(self.environment.pop_placing_agents)
        
        # Create a list of (genome_id, genome) tuples for evaluation
        genomes = list(population.population.items())
        
        # Evaluate genomes
        neat_manager.eval_genomes(genomes, config)
        
        # Find and save the best genome
        best_genome_id, best_genome = max(
            population.population.items(), key=lambda x: x[1].fitness
        )
        
        # Save the best model
        best_net = ANET(genome=best_genome, config=config) # New
        best_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=best_net,
            optimizer="adam",
            name=f"neat_best_gen_{generation}",
            lr=0.001,
        )
        best_agent.strategy.save_model(f"models/neat_gen_{generation}_best.pth")
        
        return best_genome
    
    def _update_search_agents_with_neat(self, population, config, generation):
        """Update search agents with networks from evolved NEAT genomes."""
        print("Updating search agents with evolved networks...")
        genome_items = list(population.population.items())
        
        # Make sure we don't try to update more agents than we have genomes
        for i, agent in enumerate(self.search_agents):
            if i < len(genome_items):
                genome_id, genome = genome_items[i]
                try:
                    # Store the old name before changing it
                    old_name = agent.name
                    
                    # Create new network from evolved genome
                    new_net = ANET(genome=genome, config=config) # New
                    
                    # Update the agent's network
                    agent.strategy.net = new_net
                    
                    # Update the agent's name to reflect the current generation
                    new_name = f"neat_{i}_gen_{generation+1}"
                    
                    # If the name is changing, we need to update the search_metrics
                    if old_name != new_name:
                        # Update the agent's name
                        agent.name = new_name
                        
                        # Use the SearchMetricsTracker's method to handle the name change
                        self.search_metrics.update_agent_name(old_name, new_name)
                        
                except Exception as e:
                    print(f"Error updating agent {i} with genome {genome_id}: {e}")
    
    def _evaluate_agents(self, generation):
        """Evaluate both search and placement agents against baseline opponents."""
        print(f"\nEvaluating agents against baseline opponents...")
        self.evaluator.evaluate_search_agents(self.search_agents, generation)
        self.evaluator.evaluate_placement_agents(self.environment.hof.items, generation)
    
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
            best_possible=(self.board_size**2 - sum(self.ship_sizes)),
            ylog=False,
            view=True,
        )
        visualize.plot_fitness_boxplot(stats)
        
        from neat_system.cnn_layers import global_innovation_registry
        visualize.plot_innovation_registry(global_innovation_registry)
        
        best_genomes = stats.best_genomes(5)
        genomes = [(genome, "") for genome in best_genomes]
        visualize.plot_multiple_genomes(genomes, "Best Genomes")
    
    def initialize(self):
        """Initialize the search agents and metrics tracker."""
        if self.use_neat:
            config, population, stats, weight_stats_reporter, neat_manager = self._initialize_neat()
            self.neat_config = config
            self.neat_population = population
            self.neat_stats = stats
            self.weight_stats_reporter = weight_stats_reporter
            self.neat_manager = neat_manager
            
            # Initialize search agents with NEAT genomes
            self.search_agents = self._initialize_neat_search_agents(population, config)
        else:
            # Use standard neural networks
            self.search_agents = self._initialize_nn_search_agents()
        
        # Initialize metrics tracker
        self.search_metrics = SearchMetricsTracker(self.search_agents)
    
    def run(self):
        """Run the outer evolutionary loop."""
        num_generations = self.evolution_config["evolution"]["num_generations"]

        rbuf = RBUF(max_len=self.mcts_config["replay_buffer"]["max_size"])

        if self.mcts_config["replay_buffer"]["load_from_file"]:
            rbuf.init_from_file(file_path=self.mcts_config["replay_buffer"]["file_path"])
        
        for gen in range(num_generations):
            print(f"\n=== Generation {gen + 1}/{num_generations} ===")
            
            # Evolve placing agents
            print("Evolving placing agents...")
            self.environment.evolve(gen, self.search_agents, self.search_metrics)
            
            # Train search agents with MCTS if enabled
            if self.run_mcts:
                self._train_search_agents_with_mcts(rbuf)
            
            # Handle NEAT evolution if enabled
            if self.use_neat:
                # Evaluate NEAT genomes
                self._evaluate_neat_genomes(self.neat_manager, self.neat_population, self.neat_config, gen)
                
                # Evolve the NEAT population for one generation
                self.neat_population.run(self.neat_manager.eval_genomes, 1)
                
                # Update search agents with new networks from evolved genomes
                self._update_search_agents_with_neat(self.neat_population, self.neat_config, gen)
            
            # Evaluate agents against baseline opponents
            self._evaluate_agents(gen)
        
        # Plot metrics at the end
        self._generate_visualizations()
    
    def _generate_visualizations(self):
        """Generate all visualizations at the end of training."""
        # Plot general metrics
        self.environment.plot_metrics()
        self.search_metrics.plot_metrics()
        self.evaluator.plot_metrics()
        
        # Plot NEAT-specific visualizations if NEAT was used
        if self.use_neat:
            self._visualize_neat_results(self.neat_stats, self.weight_stats_reporter, self.neat_population)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and run the outer loop manager
    outer_loop = OuterLoopManager()
    outer_loop.initialize()
    outer_loop.run()
