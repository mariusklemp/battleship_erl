import json
import os
import time

from tqdm import tqdm

from RBUF import RBUF
from deap_system.placement_ga import PlacementGeneticAlgorithm
from metrics.competitive_evaluator import CompetitiveEvaluator
from metrics.evaluator import Evaluator
from game_logic.search_agent import SearchAgent
from game_logic.game_manager import GameManager
from ai.model import ANET
from inner_loop import InnerLoopManager
from neat_system.neat_manager import NeatManager


class OuterLoopManager:
    """
    Manages the outer evolutionary loop for training Battleship agents.
    Handles both placing and search agents, with support for NEAT and standard neural networks.
    """

    def __init__(self, mcts_config_path="config/mcts_config.json",
                 evolution_config_path="config/evolution_config.json",
                 layer_config_path="ai/config_simple.json"):
        # Load configurations
        self.mcts_config = self._load_config(mcts_config_path)
        self.evolution_config = self._load_config(evolution_config_path)
        self.layer_config_path = layer_config_path

        # Extract common parameters
        self.board_size = self.mcts_config["board_size"]
        self.ship_sizes = self.mcts_config["ship_sizes"]
        self.run_inner_loop = self.mcts_config["run_inner_loop"]
        self.run_neat = self.evolution_config["run_neat"]
        self.run_ga = self.evolution_config["run_ga"]

        # Initialize components
        self.game_manager = GameManager(size=self.board_size)
        self._initialize_evaluator()
        self.placement_ga = None
        self.neat_manager = None
        self.inner_loop_manager = None

        # Initialize agents and metrics tracker
        self.search_agents = []
        self.search_agents_mapping = {}

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
        # Instantiate the CompetitiveEvaluator.
        self.competitive_evaluator = CompetitiveEvaluator(
            game_manager=self.game_manager,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            run_ga=self.run_ga,
            default_num_placement=5
        )

    def _initialize_neat(self):
        """Initialize the NEAT system for evolving neural networks."""

        self.neat_manager = NeatManager(
            neat_config_path=os.path.join("neat_system", "config.txt"),
            evolution_config=self.evolution_config,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes
        )

    def initialize_placement_ga(self):
        """Initialize the placement genetic algorithm."""

        self.placement_ga = PlacementGeneticAlgorithm(
            game_manager=self.game_manager,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            population_size=self.evolution_config["placing_population"]["size"],
            num_generations=self.evolution_config["evolution"]["num_generations"],
            elite_size=self.evolution_config["placing_population"]["elite_size"],
            MUTPB=self.evolution_config["placing_population"]["mutation_probability"],
            TOURNAMENT_SIZE=self.evolution_config["placing_population"]["tournament_size"],
        )
        self.placement_ga.initialize_placing_population(self.evolution_config["placing_population"]["size"])

    def _initialize_nn_search_agents(self):
        """Initialize search agents with neural networks."""
        population_size = self.evolution_config["search_population"]["size"]
        search_agents = []

        for i in range(population_size):
            agent = self._create_nn_agent(i * 100)
            search_agents.append(agent)

        return search_agents

    def _create_nn_agent(self, model_number):
        """Create a neural network-based search agent."""

        # Create a new network instance for each model
        net = ANET(board_size=self.board_size, activation="relu", device="cpu", layer_config=self.layer_config_path)
        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            name=f"nn_{model_number}",
            lr=0.0001,
        )
        return search_agent

    def update_search_agent_genomes(self):
        """
        Update search agents to match the current NEAT population.
        """
        population_items = self.neat_manager.population.population.items()

        # Iterate over the population and update existing agents.
        for i, (genome_id, genome) in enumerate(population_items):
            net = ANET(board_size=self.board_size, device="cpu", config=self.neat_manager.config, genome=genome)
            agent = SearchAgent(
                board_size=self.board_size,
                strategy="nn_search",
                net=net,
                optimizer="adam",
                name=f"agent_{i}",
                lr=0.0001,
            )
            self.search_agents_mapping[i] = (genome, agent)

        # Update list based on the current NEAT population.
        self.search_agents = [agent for (_, agent) in self.search_agents_mapping.values()]

    def evaluate_agents_baselines(self, search_agents, placement_agents=None, generation=0):
        """Evaluate both search and placement agents against baseline opponents."""
        self.evaluator.evaluate_search_agents(search_agents, generation)
        if placement_agents:
            self.evaluator.evaluate_placement_agents(placement_agents, generation)

    def run(self):
        """Run the outer evolutionary loop and record timings, instance counts, and memory snapshots."""

        # Data structures for logging.
        timings = {
            'baseline_evaluation': [],
            'inner_loop_training': [],
            'competitive_evaluation': [],
            'evolution': [],
        }

        search_agent_metrics = []
        # Initialize the list to hold metrics for each generation
        num_generations = self.evolution_config["evolution"]["num_generations"]
        for _ in range(num_generations):
            search_agent_metrics.append([])
        
        rbuf = RBUF()
        if self.mcts_config["replay_buffer"]["load_from_file"]:
            rbuf.init_from_file(file_path=self.mcts_config["replay_buffer"]["file_path"])

        if self.run_neat:
            self._initialize_neat()

        if self.run_ga:
            self.initialize_placement_ga()

        if self.run_inner_loop:
            self.inner_loop_manager = InnerLoopManager(game_manager=self.game_manager)

            if not self.run_neat:
                self.search_agents = self._initialize_nn_search_agents()

        for gen in range(num_generations):
            print(f"\n\n--- Generation {gen}/{num_generations} ---")
            # --- Step 1: Create/Update Search Agents (NEAT or NN) ---
            if self.run_neat:
                self.neat_manager.population.reporters.start_generation(self.neat_manager.population.generation)
                self.update_search_agent_genomes()

            # --- Step 2: Baseline Evaluation ---
            step_start = time.perf_counter()
            if self.run_ga:
                self.evaluate_agents_baselines(self.search_agents, self.placement_ga.pop_placing_agents, gen)
            else:
                self.evaluate_agents_baselines(self.search_agents, generation=gen)

            step_end = time.perf_counter()
            timings['baseline_evaluation'].append(step_end - step_start)

            # --- Step 3: Inner Loop Training ---
            step_start = time.perf_counter()
            if self.run_inner_loop:
                print(f"\nInner loop: Training {len(self.search_agents)} search agents")
                for i, search_agent in tqdm(enumerate(self.search_agents), desc="Training search agents",
                                            total=len(self.search_agents)):
                    self.inner_loop_manager.run(search_agent, rbuf=rbuf, gen=gen)

                if self.run_neat:
                    for key, (genome, search_agent) in self.search_agents_mapping.items():
                        # Transfer the trained network weights/biases + optimiser to the genome.
                        genome = search_agent.strategy.net.read_weights_biases_to_genome(genome)
                        self.search_agents_mapping[key] = (genome, search_agent)
                        self.search_agents[key] = search_agent

            step_end = time.perf_counter()
            timings['inner_loop_training'].append(step_end - step_start)

            # --- Step 4: Competitive evaluation ---
            step_start = time.perf_counter()
            if self.run_ga and self.run_neat:
                print("__Step 4: Evaluating NEAT vs GA competitively__")
                (updated_search_agents_mapping, updated_placement_agents) = self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents_mapping,
                    placement_agents=self.placement_ga.pop_placing_agents
                )
                self.placement_ga.pop_placing_agents = updated_placement_agents
                self.search_agents_mapping = updated_search_agents_mapping
            elif self.run_ga and not self.run_neat:
                print("__Step 4: Evaluating GA competitively__")
                (updated_search, updated_placement_agents) = (self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents, placement_agents=self.placement_ga.pop_placing_agents))
                self.placement_ga.pop_placing_agents = updated_placement_agents
            elif self.run_neat and not self.run_ga:
                print("__Step 4: Evaluating NEAT competitively__")
                (updated_search_agents_mapping, updated_placement_agents) = self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents_mapping,
                )
                self.search_agents_mapping = updated_search_agents_mapping

            step_end = time.perf_counter()
            timings['competitive_evaluation'].append(step_end - step_start)

            # --- Step 5: Evolution ---
            step_start = time.perf_counter()
            if self.run_neat:
                print("__Step 5: Evolving genomes__")
                self.neat_manager.evolve_one_generation()
            if self.run_ga:
                print("__Step 5: Evolving GA placement agents__")
                self.placement_ga.evolve()
            step_end = time.perf_counter()
            timings['evolution'].append(step_end - step_start)

            if not self.run_neat and not self.run_inner_loop:
                for i, search_agent in enumerate(self.search_agents):
                    if hasattr(search_agent.strategy, 'get_metrics'):
                        search_agent_metrics[gen].append(search_agent.strategy.get_metrics())
                    else:
                        print(f"Warning: Agent {i} does not have get_metrics method")
            elif self.run_inner_loop and self.run_neat:
                # Collect metrics from NEAT-based agents
                for key, (_, search_agent) in self.search_agents_mapping.items():
                    if hasattr(search_agent.strategy, 'get_metrics'):
                        search_agent_metrics[gen].append(search_agent.strategy.get_metrics())
                    else:
                        print(f"Warning: Agent {key} does not have get_metrics method")

            # --- Step 6: Save models ---
            if self.mcts_config["model"]["save"] and (gen + 1) % 10 == 0:
                model_dir = self.mcts_config["model"]["save_path"]

                if self.run_neat and self.run_inner_loop:
                    self.save_best_neat_agent(model_dir=model_dir, subdir="erl", gen=gen)
                elif self.run_neat:
                    self.save_best_neat_agent(model_dir=model_dir, subdir="neat", gen=gen)
                elif self.run_inner_loop:
                    model_path = f"{model_dir}/rl/model_gen{gen + 1}.pth"
                    self.search_agents[0].strategy.save_model(model_path)

        if not self.run_neat and not self.run_inner_loop:
            for i, search_agent in enumerate(self.search_agents):
                search_agent.strategy.plot_metrics()
        if self.run_inner_loop and self.run_neat:
            # Plot the average error and validation error of the search agents for each generation
            # Check if we have any metrics to plot
            has_metrics = any(metrics for metrics in search_agent_metrics)
            if has_metrics:
                self.plot_search_agent_metrics(search_agent_metrics)
            else:
                print("No metrics available to plot for search agents")

        # --- Step 7: Plot ---
        self._generate_visualizations(timings)

    def _generate_visualizations(self, timings):
        """Generate all visualizations at the end of training."""
        # Plot general metrics
        self.plot_timings(timings)
        self.evaluator.plot_metrics_search()
        self.competitive_evaluator.plot()

        # Plot NEAT-specific visualizations if NEAT was used
        if self.run_neat:
            self.neat_manager.visualize_results()
        if self.run_ga:
            self.evaluator.plot_metrics_placement()
            self.placement_ga.plot_metrics()

    def plot_timings(self, timings):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for label, values in timings.items():
            plt.plot(range(len(values)), values, label=label.replace("_", " ").capitalize())
        plt.xlabel("Generation")
        plt.ylabel("Time (seconds)")
        plt.title("Execution Time Per Stage")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_search_agent_metrics(self, search_agent_metrics):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Setup figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training metrics
        axs[0, 0].set_title("Training Loss")
        axs[0, 1].set_title("Validation Loss")
        axs[1, 0].set_title("Training Accuracy")
        axs[1, 1].set_title("Validation Accuracy")
        
        # Track metrics across generations
        for gen, agents_metrics in enumerate(search_agent_metrics):
            # Collect metrics from all agents in this generation
            avg_losses = []
            val_losses = []
            top1_accs = []
            top3_accs = []
            val_top1_accs = []
            val_top3_accs = []
            
            for metrics in agents_metrics:
                avg_losses.append(metrics["avg_error_history"])
                val_losses.append(metrics["avg_validation_history"])
                top1_accs.append(metrics["top1_accuracy_history"])
                top3_accs.append(metrics["top3_accuracy_history"])
                val_top1_accs.append(metrics["val_top1_accuracy_history"])
                val_top3_accs.append(metrics["val_top3_accuracy_history"])
            
            # Average metrics across all agents for this generation
            if avg_losses:
                # Plot training loss
                avg_loss = np.mean(avg_losses, axis=0)
                axs[0, 0].plot(avg_loss, label=f"Gen {gen}")
                
                # Plot validation loss
                val_loss = np.mean(val_losses, axis=0)
                axs[0, 1].plot(val_loss, label=f"Gen {gen}")
                
                # Plot training accuracy
                top1_acc = np.mean(top1_accs, axis=0)
                top3_acc = np.mean(top3_accs, axis=0)
                axs[1, 0].plot(top1_acc, label=f"Gen {gen} (Top1)")
                axs[1, 0].plot(top3_acc, linestyle='--', label=f"Gen {gen} (Top3)")
                
                # Plot validation accuracy
                val_top1_acc = np.mean(val_top1_accs, axis=0)
                val_top3_acc = np.mean(val_top3_accs, axis=0)
                axs[1, 1].plot(val_top1_acc, label=f"Gen {gen} (Top1)")
                axs[1, 1].plot(val_top3_acc, linestyle='--', label=f"Gen {gen} (Top3)")
        
        # Add labels and legends
        for ax in axs.flat:
            ax.set(xlabel='Training Iterations')
            ax.grid(alpha=0.3)
            ax.legend()
        
        axs[0, 0].set(ylabel='Loss')
        axs[0, 1].set(ylabel='Loss')
        axs[1, 0].set(ylabel='Accuracy')
        axs[1, 1].set(ylabel='Accuracy')
        
        plt.tight_layout()
        plt.savefig("search_agent_metrics.png")
        plt.show()
    def save_best_neat_agent(self, model_dir, subdir, gen):
        model_path = f"{model_dir}/{subdir}/model_gen{gen + 1}.pth"
        best_fitness = float("-inf")
        best_key = None
        best_agent = None

        for key, (genome, agent) in self.search_agents_mapping.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_key = key
                best_agent = agent

        if best_agent:
            best_agent.strategy.save_model(model_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and run the outer loop manager
    outer_loop = OuterLoopManager()
    outer_loop.run()
