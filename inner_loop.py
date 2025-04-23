import pygame
import json
import torch
import numpy as np

from RBUF import RBUF
from ai.model import ANET
from game_logic.game_manager import GameManager
import os
from game_logic.placement_agent import PlacementAgent
from ai.mcts import MCTS
from tqdm import tqdm

from game_logic.search_agent import SearchAgent
from gui import GUI
from game_logic.game_manager import GameManager
from RBUF import RBUF
from ai.model import ANET
from game_logic.search_agent import SearchAgent


class InnerLoopManager:
    def __init__(self, game_manager):
        # Load the configuration
        self.config = self.load_config("config/mcts_config.json")

        # Create game manager, MCTS, and replay buffer objects
        self.game_manager = game_manager

        self.placement_agent = PlacementAgent(
            board_size=self.config["board_size"],
            ship_sizes=self.config["ship_sizes"],
            strategy="random",
        )

        self.mcts = MCTS(
            self.game_manager,
            time_limit=self.config["mcts"]["time_limit"],
            exploration_constant=self.config["mcts"]["exploration_constant"],
        )

    @staticmethod
    def load_config(config_path="config/mcts_config.json"):
        """Load configuration from a JSON file and set the device automatically if needed."""
        with open(config_path, "r") as f:
            config = json.load(f)
        if config["device"] == "auto":
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        return config

    @staticmethod
    def canonicalize_action_distribution(action_dist: np.ndarray, board_size: int, rotation: int):
        """
        Rotate the action distribution (a flat NumPy array of length board_size^2)
        by the given number of 90° counterclockwise rotations.
        """
        dist_2d = action_dist.reshape(board_size, board_size)
        canonical_dist_2d = np.rot90(dist_2d, k=rotation)
        return canonical_dist_2d.flatten()

    def simulate_game(self, search_agent, rbuf, gui=None):
        """
        Simulate a Battleship game using the current managers and return the move count.
        If no placement agent is provided, a default random one is created.
        """

        self.placement_agent.new_placements()
        # placement_agent.show_ships()

        current_state = self.game_manager.initial_state(placing=self.placement_agent)
        self.mcts.root_node = None

        if gui:
            gui.update_board(current_state)
            pygame.display.update()

        move_count = 0
        while not self.game_manager.is_terminal(current_state):
            # visualize.show_board(current_state.board, self.game_manager.size)
            if gui:
                gui.update_board(current_state)
                pygame.display.update()

            current_node = self.mcts.run(current_state, search_agent)
            explored_ratio = sum(current_state.board[0]) / (self.game_manager.size ** 2)
            dynamic_c = self.mcts.exploration_constant * (1 - explored_ratio)
            best_child = current_node.best_child(c_param=dynamic_c)

            move = best_child.move

            # Assume state_tensor returns (canonical_board, extra_features)
            board_tensor = current_node.state.state_tensor()

            # Retrieve the raw action distribution (a NumPy array)
            action_distribution = current_node.action_distribution(board_size=self.game_manager.size)
            if action_distribution is not None:
                rbuf.add_data_point(((board_tensor), action_distribution))

            current_state = self.game_manager.next_state(current_state, move)
            move_count += 1

        return move_count

    def train_validate(self, rbuf, search_agent):
        for _ in tqdm(range(self.config["training"]["epochs"]), desc="Training epochs"):
            training_batch = rbuf.get_training_set(self.config["training"]["batch_size"])
            validation_batch = rbuf.get_validation_set()
            if len(training_batch) > 0 and len(validation_batch) > 0:
                search_agent.strategy.train_model(training_batch)
                search_agent.strategy.validate_model(validation_batch)

    def run(self, search_agent, rbuf, gen=0):
        move_counts = []
        gui = None

        if self.config["visualization"]["enable_graphics"]:
            pygame.init()
            pygame.display.set_caption("Battleship")
            gui = GUI(self.config["board_size"])

        num_games = self.config["training"]["number_actual_games"]

        if gen == 0 and self.config["model"]["save"]:
            model_path = f"{self.config['model']['save_path']}/model_gen{gen}.pth"
            search_agent.strategy.save_model(model_path)

        if self.config["training"]["play_game"]:
            for i in tqdm(range(num_games)):
                move_counts.append(self.simulate_game(search_agent, rbuf, gui))

                # Train after each game
                if self.config["model"]["train"]:
                    self.train_validate(rbuf, search_agent)

                if self.config["model"]["save"] and gen == -1 and (i + 1) % 10 == 0:
                    model_path = f"{self.config['model']['save_path']}/rl/model_gen{i + 1}.pth"
                    search_agent.strategy.save_model(model_path)

        # Train only from buffer if not playing
        elif self.config["model"]["train"]:
            self.train_validate(rbuf, search_agent)

        # Save the buffer if requested
        if self.config["replay_buffer"]["save_to_file"]:
            rbuf.save_to_file(file_path=self.config["replay_buffer"]["file_path"])


def main():
    board_size = 7

    game_manager = GameManager(board_size)

    inner_loop_manager = InnerLoopManager(game_manager)

    rbuf = RBUF(max_len=10000)

    if inner_loop_manager.config["replay_buffer"]["load_from_file"]:
        rbuf.init_from_file(file_path=inner_loop_manager.config["replay_buffer"]["file_path"])

    search_agents = []

    for i in range(1):
        net = ANET(
            board_size=board_size,
            activation="relu",
            device="cpu",
            layer_config="ai/config_simple.json",
        )

        search_agent = SearchAgent(
            board_size=board_size,
            strategy="nn_search",
            net=net,
            optimizer="adam",
            lr=0.0001,
        )
        search_agents.append(search_agent)

    for gen in range(1000):
        for i, search_agent in tqdm(enumerate(search_agents), desc="Training search agents", total=len(search_agents)):
            print(f"Training search agent {i + 1}")
            inner_loop_manager.run(search_agent, rbuf, gen=gen)

    search_agents[0].strategy.plot_metrics()


if __name__ == "__main__":
    main()
