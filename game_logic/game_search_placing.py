import sys

import pygame

from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent


class Game:
    """
    Class to manage the game state between a placing agent and a search agent
    Counts the number of moves and checks if the game is over
    """
    def __init__(self, board_size, sizes, search_strategy=None, placing_strategy=None, placing=None, search=None):
        if placing and search:
            self.placing = placing
            self.searching = search
        else:
            self.placing = PlacementAgent(board_size=board_size, sizes=sizes, strategy=placing_strategy)
            self.searching = SearchAgent(board_size=board_size, ship_sizes=sizes, strategy=search_strategy)
        self.game_over = False
        self.move_count = 0

    def play_turn(self):
        move = self.searching.strategy.find_move()

        # Update state based on the selected action
        try:
            # Check for illegal action
            assert self.is_move_possible(move), "Move has already been made"

            # Do the move
            self.make_move(move)

        except AssertionError as e:

            print("Illegal move", move)
            return False

        print(self.searching.search)
        self.move_count += 1
        # Check if all ships are sunk
        self.check_game_over()

    def is_move_possible(self, move):

        if move is None:
            return False
        if self.searching.search[0][move] != 0:
            return False
        return True

    def make_move(self, move):

        self.searching.search[0][move] = 1  # Not unknown
        if move in self.placing.indexes:
            self.searching.search[1][move] = 1  # Hit
            self.check_ship_sunk(move)
        else:
            self.searching.search[2][move] = 1  # Miss

    def check_ship_sunk(self, move):
        hit_ship = None
        sunk = True

        # Find the ship that was hit
        for ship in self.placing.ships:
            if move in ship.indexes:
                hit_ship = ship.indexes

        # Check if the ship is sunk
        for i in hit_ship:
            if self.searching.search[0][i] == 0:
                sunk = False
                break

        # If the ship is sunk, update the search board
        if sunk:
            for i in hit_ship:
                self.searching.search[3][i] = 1

    def check_game_over(self):
        all_sunk = True
        for i in self.placing.indexes:
            if self.searching.search[0][i] == 0:
                all_sunk = False
        self.game_over = all_sunk




