import json
import json
import sys
import os

import torch

from ai.mcts import MCTS
from neat_system.neat_manager import NeatManager

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
            num_variations,
            game_manager,
    ):
        self.board_size = board_size
        self.num_games = num_games
        self.num_players = num_players
        self.ship_sizes = ship_sizes
        self.placing_strategies = placing_strategies
        self.search_strategies = search_strategies
        self.game_manager = game_manager
        self.num_variations = num_variations

        self.players = {}
        self.placement_agents = {
            strat: PlacementAgent(board_size, ship_sizes, strategy=strat)
            for strat in placing_strategies
        }

        with open("../config/evolution_config.json", "r") as f:
            evolution_config = json.load(f)

        self.neat_manager = NeatManager(
            neat_config_path="../neat_system/config.txt",
            evolution_config=evolution_config,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes
        )

    def set_nn_agent(self, i, layer_config, subdir):
        """
        Load or construct an ANET‐based SearchAgent, based on whether
        the checkpoint contains a genome (NEAT/ERL) or just a plain state_dict.
        """
        model_number = i * (self.num_games // self.num_players)
        print(f"[OuterLoop] Setting up agent {model_number}")

        # Build the filesystem path
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, subdir, f"model_gen{model_number}.pth")
        assert os.path.exists(path), f"Checkpoint not found: {path}"

        # Decide if this is a NEAT checkpoint by peeking inside
        ckpt = torch.load(path, map_location="cpu")
        has_genome = isinstance(ckpt, dict) and 'genome' in ckpt

        if has_genome:
            # --- NEAT/ERL checkpoint ---
            genome = ckpt['genome']
            net = ANET(
                board_size=self.board_size,
                device="cpu",
                genome=genome,
                config=self.neat_manager.config
            )
            # Load weights from the same checkpoint
            net.load_state_dict(ckpt['model_state_dict'])
            net.eval()
        else:
            # --- Plain state_dict checkpoint ---
            net = ANET(
                board_size=self.board_size,
                activation="relu",
                device="cpu",
                layer_config=layer_config,
            )
            net.load_model(path)

        search_agent = SearchAgent(
            board_size=self.board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            name=f"nn_gen{model_number}",
            lr=0.001,
        )

        return search_agent

    def init_players(self, experiment, variation):
        self.players.clear()

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai",
                                   "config_simple.json")
        for i in range(self.num_players + 1):
            agent = self.set_nn_agent(i, config_path, f"models/5/{experiment}/{variation}")
            self.players[agent.name] = agent

    def run(self):
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=100,
            game_manager=self.game_manager,
        )
        experiments_evals = {
            "rl": [],
            "neat": [],
            "erl": [],
        }

        for experiment in experiments_evals.keys():
            print(f"\n=== {experiment.upper()} ===")
            for i in range(1, self.num_variations+1):
                print(f"\n=== Variation {i} ===")
                self.init_players(experiment=experiment, variation=i)
                for name, search_agent in tqdm(self.players.items()):
                    evaluator.evaluate_search_agents(
                        search_agents=[search_agent],
                        gen=name,
                    )
                # Append the results
                results = evaluator.search_evaluator.get_results()
                experiments_evals[experiment].append(results)
                evaluator.search_evaluator.reset()

        # 1) Aggregate each experiment’s runs
        all_stats = {
            exp: evaluator.search_evaluator.aggregate_runs(runs)
            for exp, runs in experiments_evals.items()
        }

        # 2) Plot each experiment’s avg results
        for exp, stats in all_stats.items():
            print(f"\n=== {exp.upper()} ===")
            evaluator.search_evaluator.plot_metrics_from_agg(stats, exp.upper())

        # Combined‐plot
        evaluator.search_evaluator.plot_combined_all(all_stats)

    def skill_final_agent(self, baseline=True):
        """
        Evaluate one specific trained agent (best) and the worst one,
        compare them against baselines, and plot as radar chart.
        """
        subdir = "models/5/rl"
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai", "config.json")
        agent_best = self.set_nn_agent(200, config_path, subdir)
        agent_worst = self.set_nn_agent(0, config_path, subdir)

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
            for strategy in ["mcts", "random", "hunt_down"]:
                baseline_agent = SearchAgent(
                    board_size=self.board_size,
                    strategy=strategy,
                    name=strategy,
                )
                if strategy == "mcts":
                    mcts = MCTS(self.game_manager, time_limit=1.2)
                    baseline_agent.strategy.set_mcts(mcts)
                metrics = evaluator.search_evaluator.evaluate_final_agent(baseline_agent, num_games=2)
                all_metrics.append((f"{strategy.capitalize()} Agent", metrics))

        evaluator.search_evaluator.plot_final_skill_radar_chart(all_metrics)


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                               "mcts_config.json")
    config = json.load(open(config_path))
    game_manager = GameManager(size=config["board_size"])

    tournament = Tournament(
        board_size=config["board_size"],
        num_games=100,
        ship_sizes=config["ship_sizes"],
        placing_strategies=["random", "uniform_spread"],
        search_strategies=["random", "hunt_down", "mcts"],
        num_players=10,
        num_variations=2,
        game_manager=game_manager,
    )
    # tournament.skill_final_agent(baseline=True)
    tournament.run()


if __name__ == "__main__":
    main()
