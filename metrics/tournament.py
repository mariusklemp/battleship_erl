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
            run_search=True,
            run_placement=True,
    ):
        self.board_size = board_size
        self.num_games = num_games
        self.num_players = num_players
        self.ship_sizes = ship_sizes
        self.placing_strategies = placing_strategies
        self.search_strategies = search_strategies
        self.game_manager = game_manager
        self.num_variations = num_variations
        self.run_search = run_search
        self.run_placement = run_placement

        self.search_players = {}
        self.placement_populations = {}

        with open("../config/evolution_config.json", "r") as f:
            evolution_config = json.load(f)

        self.neat_manager = NeatManager(
            neat_config_path="../neat_system/config.txt",
            evolution_config=evolution_config,
            board_size=self.board_size,
            ship_sizes=self.ship_sizes
        )

    def set_nn_agent(self, i, layer_config, subdir, best_agent=False):
        """
        Load or construct an ANET‐based SearchAgent, based on whether
        the checkpoint contains a genome (NEAT/ERL) or just a plain state_dict.
        """
        model_number = i * (self.num_games // self.num_players)
        print(f"[OuterLoop] Setting up agent {model_number}")

        # Build the filesystem path
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if best_agent:
            path = os.path.join(base, subdir, f"best_agent.pth")
        else:
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
            print("Loading plain state_dict checkpoint")
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
        self.search_players.clear()

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai",
                                   "config_simple.json")
        for i in range(self.num_players + 1):
            agent = self.set_nn_agent(i, config_path, f"models/{self.board_size}/{experiment}/{variation}")
            self.search_players[agent.name] = agent

    def init_placement_agents(self, experiment, variation):
        for i in range(self.num_players + 1):
            model_number = i * (self.num_games // self.num_players)
            file_path = f"../placement_population/{self.board_size}/{experiment}/{variation}/population_gen{model_number}.json"

            with open(file_path, "r") as f:
                chromosomes = json.load(f)

            self.placement_populations[model_number] = {}
            # Loop thorugh the chromosomes and create placement agents
            print("Loading population for:", model_number)
            for i, chromosome in enumerate(chromosomes):
                # Create a new placement agent with the genome
                print(f"Ind. {i}: {chromosome}")
                placement_agent = PlacementAgent(
                    board_size=self.board_size,
                    ship_sizes=self.ship_sizes,
                    chromosome=chromosome,
                    strategy="chromosome",
                )
                self.placement_populations[model_number][f"placing_agent{i}"] = placement_agent

    def skill_progression(self, variation=None):
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=100,
            game_manager=self.game_manager,
        )
        experiments_evals = {
            "rl": [],
            #"neat": [],
            #"erl": [],
        }
        experiment_placement = {
            "erl": []
        }

        for experiment in experiments_evals.keys():
            print(f"\n=== {experiment.upper()} ===")
            if variation:
                print(f"\n=== Variation {variation} ===")
                self.init_players(experiment=experiment, variation=variation)
                for gen, search_agent in tqdm(self.search_players.items()):
                    evaluator.evaluate_search_agents(
                        search_agents=[search_agent],
                        gen=gen,
                    )

                results = evaluator.search_evaluator.get_results()
                experiments_evals[experiment].append(results)
                evaluator.search_evaluator.reset()
            else:
                for i in range(1, self.num_variations + 1):
                    print(f"\n=== Variation {i} ===")

                    if self.run_search:
                        self.init_players(experiment=experiment, variation=i)
                        for gen, search_agent in tqdm(self.search_players.items()):
                            evaluator.evaluate_search_agents(
                                search_agents=[search_agent],
                                gen=gen,
                            )

                        results = evaluator.search_evaluator.get_results()
                        experiments_evals[experiment].append(results)
                        evaluator.search_evaluator.reset()

                    if self.run_placement and experiment in {"erl"}:
                        self.init_placement_agents(experiment=experiment, variation=i)
                        for gen, placement_agents in tqdm(self.placement_populations.items()):
                            evaluator.evaluate_placement_agents(
                                placement_agents=list(placement_agents.values()),
                                gen=gen,
                            )

                        results = evaluator.placement_evaluator.get_results()
                        experiment_placement[experiment].append(results)
                        evaluator.placement_evaluator.reset()

        # If running search tournament → aggregate & plot
        if self.run_search:
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

        if self.run_placement:
            # 1) aggregate
            all_place_stats = {
                exp: evaluator.placement_evaluator.aggregate_runs(runs)
                for exp, runs in experiment_placement.items()
            }
            # 2) per‐experiment avg plots
            for exp, stats in all_place_stats.items():
                print(f"\n=== PLACEMENT {exp.upper()} ===")
                evaluator.placement_evaluator.plot_metrics_from_agg(stats, exp.upper())

    def skill_final_agent(self, baseline=True, variation=None, experiment="rl"):
        """
        Compare average initial vs. final RL agents (across variations) + baselines
        in a single radar chart, including standard deviations.
        """
        import numpy as np
        from game_logic.search_agent import SearchAgent
        from ai.mcts import MCTS

        # Path to your NN config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ai", "config_simple.json"
        )
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.num_games // 10,
            game_manager=self.game_manager,
        )

        # 1) Gather per-variation metrics
        init_metrics = []
        fin_metrics = []
        print(f"\n=== {experiment.upper()} ===")
        if variation:
            print(f"\n=== Variation {variation} ===")
            subdir = f"models/{self.board_size}/{experiment}/{variation}"
            agent_init = self.set_nn_agent(0, config_path, subdir)
            agent_final = self.set_nn_agent(100, config_path, subdir)
            init_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_init, num_games=100))
            fin_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100))
        else:
            for var in range(1, self.num_variations + 1):
                print(f"\n=== Variation {var} ===")
                subdir = f"models/{self.board_size}/{experiment}/{var}"
                agent_init = self.set_nn_agent(0, config_path, subdir)
                agent_final = self.set_nn_agent(10, config_path, subdir)  # adjust final-gen index

                init_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_init, num_games=100))
                fin_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100))

        # 2) Compute mean & std across variations
        def mean_and_std(list_of_dicts):
            keys = list_of_dicts[0].keys()
            mean_d, std_d = {}, {}
            for k in keys:
                vals = [d[k] for d in list_of_dicts]
                mean_d[k] = float(np.mean(vals))
                std_d[k] = float(np.std(vals))
            return mean_d, std_d

        avg_init, std_init = mean_and_std(init_metrics)
        avg_fin, std_fin = mean_and_std(fin_metrics)

        # 3) Build labeled_metrics triples
        labeled_metrics = [
            (f"Avg Initial {experiment.upper()} Agent", avg_init, std_init),
            (f"Avg Final {experiment.upper()} Agent", avg_fin, std_fin),
        ]

        # 4) Append baselines (zero-std)
        if baseline:
            for strat in ("mcts", "random", "hunt_down"):
                print(f"\n=== {strat.upper()} ===")
                base = SearchAgent(board_size=self.board_size, strategy=strat, name=strat)
                if strat == "mcts":
                    m = MCTS(self.game_manager, time_limit=1.2)
                    base.strategy.set_mcts(m)
                bm = evaluator.search_evaluator.evaluate_final_agent(base, num_games=1)
                zero_std = {k: 0.0 for k in bm}
                labeled_metrics.append((f"{strat.capitalize()} Agent", bm, zero_std))

        # 5) Plot
        evaluator.search_evaluator.plot_final_skill_radar_chart(
            labeled_metrics,
            title="Skill of Final Iteration"
        )


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                               "mcts_config.json")
    config = json.load(open(config_path))
    game_manager = GameManager(size=config["board_size"])

    tournament = Tournament(
        board_size=config["board_size"],
        num_games=1000,
        ship_sizes=config["ship_sizes"],
        placing_strategies=["random", "uniform_spread"],
        search_strategies=["random", "hunt_down", "mcts"],
        num_players=10,
        num_variations=3,
        game_manager=game_manager,
        run_search=True,
        run_placement=False,
    )
    #tournament.skill_final_agent(baseline=True, variation=5, experiment="rl")
    tournament.skill_progression(variation=5)


if __name__ == "__main__":
    main()
