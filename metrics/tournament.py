import json
import json
import sys
import os

import torch

# Add the parent directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neat_system.neat_manager import NeatManager

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

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                                   "evolution_config.json")
        with open(config_path, "r") as f:
            evolution_config = json.load(f)

        neat_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neat_system",
                                        "config.txt")
        self.neat_manager = NeatManager(
            neat_config_path=neat_config_path,
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
        print(f"[OuterLoop] {i}, {self.num_games} games, {self.num_players} players")
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
            name=model_number,
            lr=0.001,
        )

        return search_agent

    def init_players(self, experiment, variation, subdir):
        self.search_players.clear()

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai",
                                   "config_simple.json")
        for i in range(self.num_players + 1):
            agent = self.set_nn_agent(i, config_path, subdir)
            self.search_players[agent.name] = agent

    def init_placement_agents(self, experiment, variation):
        for i in range(self.num_players + 1):
            model_number = i * (self.num_games // self.num_players)
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     f"placement_population/{self.board_size}/{experiment}/{variation}/population_gen{model_number}.json")

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

    def skill_progression_search(self, variation=None):
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=100,
            game_manager=self.game_manager,
        )

        experiments_evals = {
            "rl": {"solo": []},
            "neat": {"solo": [], "co_evo": []},
            "erl": {"solo": [], "co_evo": []},
        }

        for exp, modes in experiments_evals.items():
            for mode in modes:
                if exp == "rl" and mode == "co_evo":
                    continue

                print(f"\n=== SEARCH: {exp.upper()} — MODE: {mode} ===")

                if variation is not None:
                    # single-variation mode
                    print(f"Variation {variation}")
                    # pick correct folder root for this mode/experiment
                    subdir = f"models/{self.board_size}/{exp}/{mode}/{variation}"
                    self.init_players(exp, variation, subdir=subdir)
                    for gen, agent in tqdm(self.search_players.items()):
                        evaluator.evaluate_search_agents([agent], gen)
                    results = evaluator.search_evaluator.get_results()
                    experiments_evals[exp][mode].append(results)
                    evaluator.search_evaluator.reset()

                else:
                    # loop over all variations
                    for var in range(1, self.num_variations + 1):
                        print(f"\n--- Variation {var} ---")
                        # pick correct folder root for this mode/experiment
                        subdir = f"models/{self.board_size}/{exp}/{mode}/{var}"
                        self.init_players(exp, var, subdir=subdir)
                        for gen, agent in tqdm(self.search_players.items()):
                            evaluator.evaluate_search_agents([agent], gen)
                        results = evaluator.search_evaluator.get_results()
                        experiments_evals[exp][mode].append(results)
                        evaluator.search_evaluator.reset()

        # now aggregate each exp/mode separately
        all_stats = {
            exp: {
                mode: evaluator.search_evaluator.aggregate_runs(runs)
                for mode, runs in modes.items()
            }
            for exp, modes in experiments_evals.items()
        }

        # plotting
        for exp, mode_stats in all_stats.items():
            print(f"\n>>> PLOTTING {exp.upper()}")
            evaluator.search_evaluator.plot_metrics_from_agg(
                mode_stats, exp.upper()
            )

        flat_stats = {}
        for exp, stats in all_stats.items():
            if isinstance(stats, dict) and "solo" in stats:
                # unwrap the solo branch for BOTH RL (only solo) and Neat/Erl
                flat_stats[exp] = stats["solo"]
            else:
                flat_stats[exp] = stats

        evaluator.search_evaluator.plot_combined_all(flat_stats)

    def skill_progression_placement(self, variation=None):
        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=100,
            game_manager=self.game_manager,
        )
        experiment_placement = {"rl": [], "neat": [], "erl": [], "hunt_down": []}

        for experiment in experiment_placement:
            print(f"\n=== PLACEMENT: {experiment.upper()} ===")
            if variation is not None:
                self.init_placement_agents(experiment, variation)
                for gen, pop in tqdm(self.placement_populations.items()):
                    evaluator.evaluate_placement_agents(list(pop.values()), gen)
                results = evaluator.placement_evaluator.get_results()
                experiment_placement[experiment].append(results)
                evaluator.placement_evaluator.reset()
            else:
                for var in range(1, self.num_variations + 1):
                    print(f"\n--- Variation {var} ---")
                    self.init_placement_agents(experiment, var)
                    for gen, pop in tqdm(self.placement_populations.items()):
                        evaluator.evaluate_placement_agents(list(pop.values()), gen)
                    results = evaluator.placement_evaluator.get_results()
                    experiment_placement[experiment].append(results)
                    evaluator.placement_evaluator.reset()

        # aggregate & plot
        all_place_stats = {
            exp: evaluator.placement_evaluator.aggregate_runs(runs)
            for exp, runs in experiment_placement.items()
        }
        for exp, stats in all_place_stats.items():
            print(f"\n>>> PLACEMENT PLOTTING {exp.upper()}")
            evaluator.placement_evaluator.plot_metrics_from_agg(stats, exp.upper())
        evaluator.placement_evaluator.plot_combined_all(all_place_stats)

    def skill_progression(self, variation=None):
        """Legacy entry point that runs both."""
        if self.run_search:
            self.skill_progression_search(variation)
        if self.run_placement:
            self.skill_progression_placement(variation)

    def skill_final_agent(self, baseline=True, variation=None, experiment="rl"):
        """
        Compare average initial vs. final agents (across variations) + baselines
        in a single radar chart, including standard deviations.
        """
        import numpy as np
        from game_logic.search_agent import SearchAgent
        from ai.mcts import MCTS
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ai", "config_simple.json"
        )

        evaluator = Evaluator(
            board_size=self.board_size,
            ship_sizes=self.ship_sizes,
            num_evaluation_games=self.num_games,
            game_manager=self.game_manager,
        )

        init_metrics = []
        fin_metrics = []
        best_metrics = []

        print(f"\n=== {experiment.upper()} ===")
        if variation:
            print(f"\n=== Variation {variation} ===")
            subdir = f"models/{self.board_size}/{experiment}/{variation}"
            agent_init = self.set_nn_agent(0, config_path, subdir)
            agent_final = self.set_nn_agent(10, config_path, subdir)
            init_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_init, num_games=100))
            fin_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100))

            if experiment != "rl":
                agent_best = self.set_nn_agent(10, config_path, subdir, best_agent=True)
                best_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_best, num_games=100))

        else:
            for var in range(1, self.num_variations + 1):
                print(f"\n=== Variation {var} ===")
                subdir = f"models/{self.board_size}/{experiment}/solo/{var}"
                agent_init = self.set_nn_agent(0, config_path, subdir)
                agent_final = self.set_nn_agent(10, config_path, subdir)
                init_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_init, num_games=100))
                fin_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100))

                if experiment != "rl":
                    agent_best = self.set_nn_agent(10, config_path, subdir, best_agent=True)
                    best_metrics.append(evaluator.search_evaluator.evaluate_final_agent(agent_best, num_games=100))

        # --- Aggregate mean and std
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

        labeled_metrics = [
            (f"Avg Initial {experiment.upper()} Agent", avg_init, std_init),
            (f"Avg Final {experiment.upper()} Agent", avg_fin, std_fin),
        ]

        if experiment != "rl":
            avg_best, std_best = mean_and_std(best_metrics)
            labeled_metrics.append((f"Best Final {experiment.upper()} Agent", avg_best, std_best))

        # --- Append baselines properly with deviation ---
        if baseline:
            # 1) gather per‐replicate metrics
            baseline_results = {strat: [] for strat in ("mcts", "random", "hunt_down")}
            for strat in baseline_results:
                print(f"\n=== {strat.upper()} ===")
                base = SearchAgent(board_size=self.board_size, strategy=strat, name=strat)
                if strat == "mcts":
                    m = MCTS(self.game_manager, time_limit=1.2)
                    base.strategy.set_mcts(m)
                for rep in range(self.num_variations):
                    bm = evaluator.search_evaluator.evaluate_final_agent(base, num_games=2)
                    baseline_results[strat].append(bm)

            # 2) collapse across replicates
            for strat, results in baseline_results.items():
                avg_bm, std_bm = mean_and_std(results)
                labeled_metrics.append((f"{strat.capitalize()} Agent", avg_bm, std_bm))

        evaluator.search_evaluator.plot_final_skill_radar_chart(
            labeled_metrics,
            title="Skill of Final Iteration"
        )

    def skill_final_agent_combined(self, baseline=True):
        """
        Compare average final agents across RL/NEAT/ERL variations + baselines
        in a single radar chart, including standard deviations.
        """
        import numpy as np
        import os
        from game_logic.search_agent import SearchAgent
        from ai.mcts import MCTS

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

        experiments = {"rl": [], "neat": [], "erl": []}

        # 1) Gather per-variation metrics
        for exp in experiments:
            print(f"\n=== {exp.upper()} ===")
            for var in range(1, self.num_variations + 1):
                print(f"--- Variation {var} ---")
                subdir = f"models/{self.board_size}/{exp}/solo/{var}"
                agent_final = self.set_nn_agent(10, config_path, subdir)
                metrics = evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100)
                experiments[exp].append(metrics)

        # 2) Compute mean & std across variations
        def mean_and_std(list_of_dicts):
            keys = list_of_dicts[0].keys()
            mean_d, std_d = {}, {}
            for k in keys:
                vals = [d[k] for d in list_of_dicts]
                mean_d[k] = float(np.mean(vals))
                std_d[k] = float(np.std(vals))
            return mean_d, std_d

        # 3) Build labeled_metrics triples
        labeled_metrics = []
        for exp in ("erl", "neat", "rl"):
            avg_fin, std_fin = mean_and_std(experiments[exp])
            labeled_metrics.append((f"Final {exp.upper()} Agent", avg_fin, std_fin))

        # 4) Append baselines with std (same as in skill_final_agent)
        if baseline:
            baseline_results = {strat: [] for strat in ("mcts", "random", "hunt_down")}
            for strat in baseline_results:
                print(f"\n=== {strat.upper()} ===")
                base = SearchAgent(board_size=self.board_size, strategy=strat, name=strat)
                if strat == "mcts":
                    m = MCTS(self.game_manager, time_limit=1.2)
                    base.strategy.set_mcts(m)
                for rep in range(self.num_variations):
                    bm = evaluator.search_evaluator.evaluate_final_agent(base, num_games=2)
                    baseline_results[strat].append(bm)

            for strat, results in baseline_results.items():
                avg_bm, std_bm = mean_and_std(results)
                labeled_metrics.append((f"{strat.capitalize()} Agent", avg_bm, std_bm))

        # 5) Plot
        evaluator.search_evaluator.plot_final_skill_radar_chart(
            labeled_metrics,
            title="Skill of Final Iteration"
        )

    def skill_final_agent_combined(self, baseline=True):
        """
        Compare average final agents across RL/NEAT/ERL variations + baselines
        in a single radar chart, including standard deviations.
        """
        import numpy as np
        import os
        from game_logic.search_agent import SearchAgent
        from ai.mcts import MCTS

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

        experiments = {"rl": [], "neat": [], "erl": []}

        # 1) Gather per-variation metrics
        for exp in experiments:
            print(f"\n=== {exp.upper()} ===")
            for var in range(1, self.num_variations + 1):
                print(f"--- Variation {var} ---")
                subdir = f"models/{self.board_size}/{exp}/solo/{var}"
                agent_final = self.set_nn_agent(10, config_path, subdir)
                metrics = evaluator.search_evaluator.evaluate_final_agent(agent_final, num_games=100)
                experiments[exp].append(metrics)

        # 2) Compute mean & std across variations
        def mean_and_std(list_of_dicts):
            keys = list_of_dicts[0].keys()
            mean_d, std_d = {}, {}
            for k in keys:
                vals = [d[k] for d in list_of_dicts]
                mean_d[k] = float(np.mean(vals))
                std_d[k] = float(np.std(vals))
            return mean_d, std_d

        # 3) Build labeled_metrics triples
        labeled_metrics = []
        for exp in ("erl", "neat", "rl"):
            avg_fin, std_fin = mean_and_std(experiments[exp])
            labeled_metrics.append((f"Final {exp.upper()} Agent", avg_fin, std_fin))

        # 4) Append baselines (zero-std)
        if baseline:
            for strat in ("mcts", "random", "hunt_down"):
                print(f"\n=== {strat.upper()} ===")
                base = SearchAgent(board_size=self.board_size, strategy=strat, name=strat)
                if strat == "mcts":
                    m = MCTS(self.game_manager, time_limit=1.2)
                    base.strategy.set_mcts(m)
                bm = evaluator.search_evaluator.evaluate_final_agent(base, num_games=10)
                zero_std = {k: 0.0 for k in bm}
                labeled_metrics.append((f"{strat.capitalize()} Agent", bm, zero_std))

        # 5) Plot
        evaluator.search_evaluator.plot_final_skill_radar_chart(
            labeled_metrics,
            title="Skill of Final Iteration"
        )


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config", "mcts_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    game_manager = GameManager(size=config["board_size"])

    tournament = Tournament(
        board_size=config["board_size"],
        num_games=100,
        ship_sizes=config["ship_sizes"],
        placing_strategies=["random", "uniform_spread"],
        search_strategies=["random", "hunt_down", "mcts"],
        num_players=10,
        num_variations=5,
        game_manager=game_manager,
        run_search=True,
        run_placement=False,
    )
    #tournament.skill_final_agent(baseline=True, experiment="rl")
    tournament.skill_progression()
    #tournament.skill_final_agent_combined(baseline=True)


if __name__ == "__main__":
    main()
