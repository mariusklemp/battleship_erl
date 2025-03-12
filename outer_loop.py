import json
import os
from tqdm import tqdm
import neat
import configparser
from deap_system.evolution import Evolution
from deap_system.evolution_evaluator import EvolutionEvaluator
from game_logic.search_agent import SearchAgent
from game_logic.game_manager import GameManager
from ai.models import ANET
from deap_system.search_metrics import SearchMetricsTracker
from run_mcts import run_mcts_inner_loop, load_config
from neat_system.neat_manager import NEAT_Manager
from neat_system.cnn_genome import CNNGenome
import visualize
from neat_system.weight_reporter import WeightStatsReporter


def load_evolution_config(config_path="config/evolution_config.json"):
    """Load evolution configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def set_nn_agent(i, layer_config, board_size, model_number):
    """Initializes a SearchAgent that uses a neural network.
    Model number is based on tournament parameters.
    """
    # Calculate model number to load models at regular intervals (0, 100, 200, ..., 1000)
    print(f"[DEBUG] Loading model: {model_number}")
    path = f"models/model_{model_number}.pth"

    # Create a new network instance for each model
    net = ANET(
        board_size=board_size,
        activation="relu",
        output_size=board_size**2,
        device="cpu",
        layer_config=layer_config,
    )

    search_agent = SearchAgent(
        board_size=board_size,
        strategy="nn_search",
        net=net,
        optimizer="adam",
        name=f"nn_{model_number}",
        lr=0.001,
    )
    # search_agent.strategy.load_model(path)
    return search_agent


def populate_search_agents(board_size, population_size, layer_config):
    """
    Populate the search agents.
    """
    search_agents = []
    for i in range(population_size + 1):
        agent = set_nn_agent(i, layer_config, board_size, i * 100)
        search_agents.append(agent)
    return search_agents


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Load configurations
    mcts_config = load_config()
    evolution_config = load_evolution_config()
    RUN_MCTS = True
    USE_NEAT = True

    # Load neural network configuration
    LAYER_CONFIG = json.load(open("ai/config.json"))

    # Get configuration parameters
    BOARD_SIZE = mcts_config["board_size"]
    SHIP_SIZES = mcts_config["ship_sizes"]
    PLACING_POPULATION_SIZE = evolution_config["placing_population"]["size"]
    SEARCH_POPULATION_SIZE = evolution_config["search_population"]["size"]
    ELITE_SIZE = evolution_config["placing_population"]["elite_size"]
    NUM_GENERATIONS = evolution_config["evolution"]["num_generations"]
    TOURNAMENT_SIZE = evolution_config["placing_population"]["tournament_size"]
    MUTPB = evolution_config["placing_population"]["mutation_probability"]
    INNER_LOOP_EPOCHS = evolution_config["evolution"]["inner_loop_epochs"]
    # Number of games to run for each evaluation
    NUM_EVALUATION_GAMES = 10

    # Initialize the evolution environment for placing agents
    environment = Evolution(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        population_size=PLACING_POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        elite_size=ELITE_SIZE,
        MUTPB=MUTPB,
        TOURNAMENT_SIZE=TOURNAMENT_SIZE,
    )

    # Create a game manager
    game_manager = GameManager(size=BOARD_SIZE)

    if USE_NEAT:
        # Set up NEAT configuration
        neat_config_path = os.path.join("neat_system", "config.txt")

        # Update the config file with current parameters
        cp = configparser.ConfigParser()
        cp.read(neat_config_path)

        # Update NEAT configuration
        cp["NEAT"]["fitness_threshold"] = str(BOARD_SIZE**2 - sum(SHIP_SIZES))
        cp["NEAT"]["pop_size"] = str(SEARCH_POPULATION_SIZE)

        # Update CNNGenome configuration
        cp["CNNGenome"]["input_size"] = str(BOARD_SIZE)
        cp["CNNGenome"]["output_size"] = str(BOARD_SIZE**2)

        # Write updated config back to disk
        with open(neat_config_path, "w") as f:
            cp.write(f)

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
        RANGE_EVALUATIONS = 5  # Number of games to evaluate each genome

        neat_manager = NEAT_Manager(
            board_size=BOARD_SIZE,
            ship_sizes=SHIP_SIZES,
            strategy_placement="random",
            strategy_search="nn_search",
            range_evaluations=RANGE_EVALUATIONS,
            config=config,
            game_manager=game_manager,
        )

        # Create initial search agents from the NEAT population
        search_agents = []
        for i, (genome_id, genome) in enumerate(population.population.items()):
            try:
                net = ANET.create_from_cnn_genome(genome=genome, config=config)
                agent = SearchAgent(
                    board_size=BOARD_SIZE,
                    strategy="nn_search",
                    net=net,
                    optimizer="adam",
                    name=f"neat_{i}_gen_0",  # Use generation-based naming from the start
                    lr=0.001,
                )
                search_agents.append(agent)
            except Exception as e:
                print(f"Error creating agent for genome {genome_id}: {e}")
    else:
        # Use the original approach with fixed neural networks
        search_agents = populate_search_agents(
            BOARD_SIZE, SEARCH_POPULATION_SIZE, LAYER_CONFIG
        )

    # Initialize metrics tracker
    search_metrics = SearchMetricsTracker(search_agents)

    # Initialize the evolution evaluator
    evaluator = EvolutionEvaluator(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        num_evaluation_games=NUM_EVALUATION_GAMES,
    )

    # Create directory for saving models
    os.makedirs("models", exist_ok=True)

    # ---------------------------------------------------------------------
    # Outer loop for generations
    # ---------------------------------------------------------------------
    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generation {gen + 1}/{NUM_GENERATIONS} ===")

        # Evolve placing agents
        print("Evolving placing agents...")
        environment.evolve(gen, search_agents, search_metrics)

        # Train search agents with MCTS
        if RUN_MCTS:
            print("\nTraining search agents against placing population...")
            for i, search_agent in tqdm(
                enumerate(search_agents),
                desc="Training search agents",
                total=len(search_agents),
            ):
                print(f"\nTraining search agent {i + 1}/{len(search_agents)}")
                for _ in range(INNER_LOOP_EPOCHS):
                    run_mcts_inner_loop(
                        environment.game_manager,
                        search_agent,
                        simulations_number=mcts_config["mcts"]["simulations_number"],
                        exploration_constant=mcts_config["mcts"]["exploration_constant"],
                        batch_size=mcts_config["training"]["batch_size"],
                        device=mcts_config["device"],
                        sizes=SHIP_SIZES,
                        placement_agents=environment.pop_placing_agents,
                        epochs=mcts_config["training"]["epochs"],
                    )

        if USE_NEAT:
            # Evaluate NEAT genomes
            print("Evaluating NEAT genomes...")
            neat_manager.set_placement_agents(environment.pop_placing_agents)
            # Create a list of (genome_id, genome) tuples for evaluation
            genomes = list(population.population.items())

            # Evaluate genomes using the eval_genomes method
            neat_manager.eval_genomes(genomes, config)

            # Find the best genome
            best_genome_id, best_genome = max(
                population.population.items(), key=lambda x: x[1].fitness
            )

            # Save the best model
            best_net = ANET.create_from_cnn_genome(genome=best_genome, config=config)
            best_agent = SearchAgent(
                board_size=BOARD_SIZE,
                strategy="nn_search",
                net=best_net,
                optimizer="adam",
                name=f"neat_best_gen_{gen}",  # Use consistent generation-based naming
                lr=0.001,
            )
            best_agent.strategy.save_model(f"models/neat_gen_{gen}_best.pth")

            # Evolve the NEAT population for one generation
            population.run(neat_manager.eval_genomes, 1)

            # Update existing search agents with new networks from evolved genomes
            print("Updating search agents with evolved networks...")
            genome_items = list(population.population.items())

            # Make sure we don't try to update more agents than we have genomes
            for i, agent in enumerate(search_agents):
                if i < len(genome_items):
                    genome_id, genome = genome_items[i]
                    try:
                        # Store the old name before changing it
                        old_name = agent.name

                        # Create new network from evolved genome
                        new_net = ANET.create_from_cnn_genome(
                            genome=genome, config=config
                        )

                        # Update the agent's network
                        agent.strategy.net = new_net

                        # Update the agent's name to reflect the current generation
                        new_name = f"neat_{i}_gen_{gen+1}"

                        # If the name is changing, we need to update the search_metrics
                        if old_name != new_name:
                            # Update the agent's name
                            agent.name = new_name

                            # Use the SearchMetricsTracker's method to handle the name change
                            search_metrics.update_agent_name(old_name, new_name)

                    except Exception as e:
                        print(f"Error updating agent {i} with genome {genome_id}: {e}")
                        # If updating fails, we keep the old network

        # Evaluate agents against baseline opponents
        print(f"\nEvaluating agents against baseline opponents...")
        evaluator.evaluate_search_agents(search_agents, gen)
        evaluator.evaluate_placement_agents(environment.hof.items, gen)

    # Plot metrics at the end
    environment.plot_metrics()
    search_metrics.plot_metrics()
    evaluator.plot_metrics()

    # Plot NEAT statistics if NEAT was used
    if USE_NEAT:
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
            best_possible=(BOARD_SIZE**2 - sum(SHIP_SIZES)),
            ylog=False,
            view=True,
        )
        visualize.plot_fitness_boxplot(stats)
        from neat_system.cnn_layers import global_innovation_registry

        visualize.plot_innovation_registry(global_innovation_registry)

        best_genomes = stats.best_genomes(5)
        genomes = []
        for genome in best_genomes:
            genomes.append((genome, ""))
        visualize.plot_multiple_genomes(genomes, "Best Genomes")
