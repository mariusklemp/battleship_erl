import json
import json
import sys
import os

# Add the parent directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from game_logic.game_manager import GameManager
from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent
from ai.model import ANET
from evaluator import Evaluator


class Tournament:
    def __init__(
            self,
            board_size,
            num_games,
            ship_sizes,
            placing_strategies,
            search_strategies,
            num_players,
            game_manager,
    ):
        self.board_size = board_size
        self.num_games = num_games
        self.num_players = num_players
        self.ship_sizes = ship_sizes
        self.placing_strategies = placing_strategies
        self.search_strategies = search_strategies
        self.game_manager = game_manager

        self.players = {}
        self.placement_agents = {
            strat: PlacementAgent(board_size, ship_sizes, strategy=strat)
            for strat in placing_strategies
        }

    def set_nn_agent(self, i, layer_config):
        model_number = i * (self.num_games // self.num_players)
        print(f"Setting up agent {model_number}")

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/5/rl",
                            f"model_gen{model_number}.pth")

        net = ANET(
            board_size=self.board_size,
            activation="relu",
            device="cpu",
            layer_config=layer_config,
        )

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            name=model_number,
            lr=0.001,
        )
        search_agent.strategy.load_model(path)
        return search_agent

    def init_players(self, time_limit):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai", "config.json")
        for i in range(self.num_players + 1):
            agent = self.set_nn_agent(i, config_path)
            self.players[agent.name] = agent

    def run(self):
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.num_games // 10,
            game_manager=self.game_manager,
        )

        for name, search_agent in tqdm(self.players.items()):
            evaluator.evaluate_search_agents(
                search_agents=[search_agent],
                gen=name,
            )

        evaluator.plot_metrics_search()

    def skill_final_agent(self, agent_index=0, baseline=True):
        """
        Evaluate one specific trained agent (best) and the worst one,
        compare them against baselines, and plot as radar chart.
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai", "config.json")
        agent_best = self.set_nn_agent(agent_index, config_path)
        agent_worst = self.set_nn_agent(0, config_path)

        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.num_games // 10,
            game_manager=self.game_manager,
        )

        all_metrics = [
            ("Best Agent", evaluator.search_evaluator.evaluate_final_agent(agent_best, num_games=100)),
            ("Worst Agent", evaluator.search_evaluator.evaluate_final_agent(agent_worst, num_games=100)),
        ]

        if baseline:
            for strategy in ["random", "hunt_down"]:
                baseline_agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strategy,
                    name=strategy,
                )
                metrics = evaluator.search_evaluator.evaluate_final_agent(baseline_agent, num_games=100)
                all_metrics.append((f"{strategy.capitalize()} Agent", metrics))

        evaluator.search_evaluator.plot_final_skill_radar_chart(all_metrics)


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                               "mcts_config.json")
    config = json.load(open(config_path))
    game_manager = GameManager(size=config["board_size"])

    tournament = Tournament(
        board_size=config["board_size"],
        num_games=config["training"]["number_actual_games"],
        ship_sizes=config["ship_sizes"],
        placing_strategies=["random", "uniform_spread"],
        search_strategies=["random", "hunt_down", "mcts"],
        num_players=10,
        game_manager=game_manager,
    )
    # tournament.init_players(time_limit=config["mcts"]["time_limit"])
    tournament.skill_final_agent(agent_index=200, baseline=True)
    # tournament.run()


if __name__ == "__main__":
    main()
