import json
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
        path = f"../models/model_{model_number}.pth"

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
        layer_config = json.load(open("../ai/config.json"))

        for i in range(self.num_players + 1):
            agent = self.set_nn_agent(i, layer_config)
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


def main():
    config = json.load(open("../config/mcts_config.json"))
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
    tournament.init_players(time_limit=config["mcts"]["time_limit"])
    tournament.run()


if __name__ == "__main__":
    main()
