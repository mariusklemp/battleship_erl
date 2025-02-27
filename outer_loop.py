import json
from tqdm import tqdm
from deap_system.evolution import Evolution
from deap_system.evolution_evaluator import EvolutionEvaluator
from game_logic.search_agent import SearchAgent
from ai.models import ANET
from deap_system.search_metrics import SearchMetricsTracker
from run_mcts import run_mcts_inner_loop, load_config


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

    # Number of games to run for each evaluation
    NUM_EVALUATION_GAMES = 10

    environment = Evolution(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        population_size=PLACING_POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        elite_size=ELITE_SIZE,
        MUTPB=MUTPB,
        TOURNAMENT_SIZE=TOURNAMENT_SIZE,
    )

    # Initialize search agents and metrics tracker
    search_agents = populate_search_agents(
        BOARD_SIZE, SEARCH_POPULATION_SIZE, LAYER_CONFIG
    )
    search_metrics = SearchMetricsTracker(search_agents)

    # Initialize the evolution evaluator for tracking performance against baseline opponents
    evaluator = EvolutionEvaluator(
        board_size=BOARD_SIZE,
        ship_sizes=SHIP_SIZES,
        num_evaluation_games=NUM_EVALUATION_GAMES,
    )

    # Outer loop for generations of placing agents and search agents
    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generation {gen + 1}/{NUM_GENERATIONS} ===")

        # Run evolution with metrics tracking
        print("Evolving placing agents...")
        environment.evolve(gen, search_agents, search_metrics)

        # Run MCTS training of search agents in inner loop
        print("\nTraining search agents against placing population...")
        if RUN_MCTS:
            for i, search_agent in tqdm(
                enumerate(search_agents),
                desc="Training search agents",
                total=len(search_agents),
            ):
                print(
                    f"\nTraining search agent {i + 1}/{len(search_agents)} against {PLACING_POPULATION_SIZE} placing agents"
                )
                run_mcts_inner_loop(
                    environment.game_manager,
                    search_agent,
                    simulations_number=mcts_config["mcts"]["simulations_number"],
                    exploration_constant=mcts_config["mcts"]["exploration_constant"],
                    batch_size=mcts_config["training"]["batch_size"],
                    device=mcts_config["device"],
                    sizes=SHIP_SIZES,
                    placement_agents=environment.pop_placing_agents,
                )

        # Evaluate agents against baseline opponents after each generation
        print(
            f"\nEvaluating agents against baseline opponents ({NUM_EVALUATION_GAMES} games each)..."
        )
        evaluator.evaluate_search_agents(search_agents, gen)
        evaluator.evaluate_placement_agents(environment.hof.items, gen)

    # Plot metrics at the end
    environment.plot_metrics()
    search_metrics.plot_metrics()

    # Plot evolution evaluation metrics
    print("\nPlotting evolution evaluation metrics...")
    evaluator.plot_metrics()
