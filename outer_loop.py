import json
import os

from tqdm import tqdm

from rl.RBUF import RBUF
from deap_system.placement_ga import PlacementGeneticAlgorithm
from competitive_evaluator import CompetitiveEvaluator
from metrics.evaluator import Evaluator
from game_logic.search_agent import SearchAgent
from game_logic.game_manager import GameManager
from rl.model import ANET
from inner_loop import InnerLoopManager
from neat_system.neat_manager import NeatManager


class OuterLoopManager:
    """
    Manages the outer evolutionary loop for training Battleship agents.
    Handles both placing and search agents, with support for NEAT and standard neural networks.
    """

    def __init__(self, mcts_config_path="config/mcts_config.json",
                 evolution_config_path="config/evolution_config.json",
                 layer_config_path="config/cnn_config.json"):
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
        self.device = self.mcts_config["device"]

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
            default_num_placement=self.evolution_config["placing_population"]["size"],
        )

    def _initialize_neat(self):
        """Initialize the NEAT system for evolving neural networks."""

        self.neat_manager = NeatManager(
            neat_config_path=os.path.join("config", "neat_config.txt"),
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

    def _initialize_hunt_down_search_agents(self):
        """Initialize search agents with neural networks."""
        population_size = self.evolution_config["search_population"]["size"]
        search_agents = []

        for i in range(population_size):
            agent = SearchAgent(
                board_size=self.board_size,
                strategy="hunt_down",
                name=f"hunt_down_{i}",
            )

            search_agents.append(agent)

        return search_agents

    def _create_nn_agent(self, model_number):
        """Create a neural network-based search agent."""

        # Create a new network instance for each model
        net = ANET(board_size=self.board_size, activation="relu", device=self.device,
                   layer_config=self.layer_config_path)
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
            net = ANET(board_size=self.board_size, device=self.device, config=self.neat_manager.config, genome=genome)
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

        num_generations = self.evolution_config["evolution"]["num_generations"]
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

        if not self.run_neat and not self.run_inner_loop:
            self.search_agents = self._initialize_hunt_down_search_agents()

        for gen in range(num_generations):
            print(f"\n\n--- Generation {gen}/{num_generations} ---")
            # --- Step 1: Create/Update Search Agents from genomes ---
            if self.run_neat:
                self.neat_manager.population.reporters.start_generation(self.neat_manager.population.generation)
                self.update_search_agent_genomes()

            # --- Step 2: Save initial models ---
            if self.mcts_config["model"]["save"] and gen == 0:
                self._save_models(gen, initial=True)

            # --- Step 3: Baseline Evaluation ---
            if self.run_ga:
                self.evaluate_agents_baselines(self.search_agents, self.placement_ga.pop_placing_agents, gen)
            else:
                self.evaluate_agents_baselines(self.search_agents, generation=gen)

            # --- Step 4: Inner Loop (RL) Training ---
            if self.run_inner_loop:
                for i, search_agent in tqdm(enumerate(self.search_agents), desc="Training search agents",
                                            total=len(self.search_agents)):
                    self.inner_loop_manager.run(search_agent, rbuf=rbuf, gen=gen)

                if self.run_neat:
                    for key, (genome, search_agent) in self.search_agents_mapping.items():
                        # Transfer the trained network weights/biases + optimiser to the genome.
                        genome = search_agent.strategy.net.read_weights_biases_to_genome(genome)
                        self.search_agents_mapping[key] = (genome, search_agent)
                        self.search_agents[key] = search_agent

            # --- Step 5: Competitive evaluation ---
            if self.run_ga and self.run_neat:
                (updated_search_agents_mapping, updated_placement_agents) = self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents_mapping,
                    placement_agents=self.placement_ga.pop_placing_agents
                )
                self.placement_ga.pop_placing_agents = updated_placement_agents
                self.search_agents_mapping = updated_search_agents_mapping
            elif self.run_ga and not self.run_neat:
                (updated_search, updated_placement_agents) = (self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents, placement_agents=self.placement_ga.pop_placing_agents))
                self.placement_ga.pop_placing_agents = updated_placement_agents
            elif self.run_neat and not self.run_ga:
                (updated_search_agents_mapping, updated_placement_agents) = self.competitive_evaluator.evaluate(
                    search_agents=self.search_agents_mapping,
                )
                self.search_agents_mapping = updated_search_agents_mapping

            # --- Step 6: Evolution ---
            if self.run_neat:
                self.neat_manager.evolve_one_generation()
            if self.run_ga:
                self.placement_ga.evolve()

            # --- Step 7: Save models ---
            if self.mcts_config["model"]["save"] and (gen + 1) % 10 == 0:
                self._save_models(gen)

        # --- Step 8: Plot ---
        self._generate_visualizations()

    def _generate_visualizations(self):
        """Generate all visualizations at the end of training."""
        # Plot general metrics
        self.evaluator.plot_metrics_search()

        # Plot RL metrics if only RL was used
        if not self.run_neat and self.run_inner_loop:
            for i, search_agent in enumerate(self.search_agents):
                search_agent.strategy.plot_metrics()

        # Plot NEAT-specific visualizations if NEAT was used
        if self.run_neat:
            self.neat_manager.visualize_results()

        if self.run_ga:
            self.evaluator.plot_metrics_placement()
            self.competitive_evaluator.plot_hall_frequencies(
                hof_per_generation=self.placement_ga.hof_per_generation,
                hos_per_generation=self.placement_ga.hos_per_generation,
                board_size=self.placement_ga.board_size
            )
            self.competitive_evaluator.plot_halls_wrapped(
                hof_per_generation=self.placement_ga.hof_per_generation,
                hos_per_generation=self.placement_ga.hos_per_generation,
            )

        self.competitive_evaluator.plot()

    def _save_models(self, gen: int, initial: bool = False):
        """
        Save both search‐ and placement‐models based on current flags.
        If `initial` is True, uses the 'initial' naming convention/gen offset.
        """
        model_dir_search    = f"models/{self.board_size}"
        model_dir_placement = f"placement_population/{self.board_size}"
        variant             = self.evolution_config["variant"]

        # pick subdirectory name based on modes
        if   self.run_neat and self.run_inner_loop: subdir = "erl"
        elif self.run_neat:                          subdir = "neat"
        elif self.run_inner_loop:                    subdir = "rl"
        else:                                        subdir = "hunt_down"

        # For NEAT we always call save_best_neat_agent()
        if self.run_neat:
            save_gen = gen - 1 if initial else gen
            self.save_best_neat_agent(
                model_dir=model_dir_search,
                subdir=subdir,
                gen=save_gen,
                variant=variant
            )
        # For pure RL (inner loop) we save the raw .pth
        elif self.run_inner_loop:
            # initial saves use `model_gen{gen}.pth`, periodic use `model_gen{gen+1}.pth`
            offset   = 0 if initial else 1
            save_gen = gen + offset
            model_path = (
                f"{model_dir_search}/"
                f"{subdir}/solo/{variant}/"
                f"model_gen{save_gen}.pth"
            )
            self.search_agents[0].strategy.net.save_model(model_path)

        # For GA we always dump the placement population
        if self.run_ga:
            save_gen = gen - 1 if initial else gen
            self.save_placement_population(
                chromosomes=self.placement_ga.population_chromosomes,
                model_dir=model_dir_placement,
                subdir=subdir,
                gen=save_gen,
                variant=variant
            )

    def save_best_neat_agent(self, model_dir, subdir, gen, variant):
        if self.run_ga:
            model_path = f"{model_dir}/{subdir}/co_evo/{variant}/model_gen{gen + 1}.pth"
        else:
            model_path = f"{model_dir}/{subdir}/solo/{variant}/model_gen{gen + 1}.pth"
        best_fitness = float("-inf")
        best_agent = None
        if gen == -1:
            best_agent = self.search_agents_mapping[0][1]
            best_agent.strategy.net.save_model_genome(model_path)
            return

        for key, (genome, agent) in self.search_agents_mapping.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_agent = agent

        if best_agent:
            best_agent.strategy.net.save_model_genome(model_path)

    def save_placement_population(self, chromosomes, model_dir, subdir, gen, variant):
        """
        Saves the list of placement chromosomes to a JSON file.
        Each chromosome is a list of ship placements.
        """
        file_path = f"{model_dir}/{subdir}/{variant}/population_gen{gen + 1}.json"

        with open(file_path, "w") as f:
            json.dump(chromosomes, f, indent=2)

        print(f"✅ Saved placement chromosomes to {file_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    outer_loop = OuterLoopManager()
    outer_loop.run()
