import json
from deapSystem.evolution import Evolution
from game_logic.search_agent import SearchAgent
from ai.models import ANET
from deapSystem.search_metrics import SearchMetricsTracker


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
    search_agent.strategy.load_model(path)
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
    # === Static Parameters (Adjustable) ===
    BOARD_SIZE = 5
    SHIP_SIZES = [3, 2, 2]
    PLACING_POPULATION_SIZE = 50
    SEARCH_POPULATION_SIZE = 10
    ELITE_SIZE = 1
    NUM_GENERATIONS = 10
    TOURNAMENT_SIZE = 3
    MUTPB = 0.2
    LAYER_CONFIG = json.load(open("ai/config.json"))
    # =======================================

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

    for gen in range(NUM_GENERATIONS):
        # Run evolution with metrics tracking
        environment.evolve(gen, search_agents, search_metrics)

    # Plot metrics at the end
    environment.plot_metrics()
    search_metrics.plot_metrics()
