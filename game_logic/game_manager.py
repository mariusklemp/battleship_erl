import sys

import pygame

from game_logic.placement_agent import PlacementAgent
from game_logic.search_agent import SearchAgent


class Game:
    """
    Class to manage the game state between two players
    """

    def __init__(
        self,
        board_size,
        sizes,
        player1_search_strategy,
        player1_placing_strategy,
        player2_search_strategy,
        player2_placing_strategy,
    ):
        self.player1Placement = PlacementAgent(
            board_size=board_size, ship_sizes=sizes, strategy=player1_placing_strategy
        )
        self.player1Search = SearchAgent(
            board_size=board_size, ship_sizes=sizes, strategy=player1_search_strategy
        )
        self.player2Placement = PlacementAgent(
            board_size=board_size, ship_sizes=sizes, strategy=player2_placing_strategy
        )
        self.player2Search = SearchAgent(
            board_size=board_size, ship_sizes=sizes, strategy=player2_search_strategy
        )
        self.player1_turn = True
        self.game_over = False
        self.result = None

    def play_turn(self, gui=None):
        move = None

        if self.player1_turn:  # (Player1)
            if gui:  # Human player move
                for event in pygame.event.get():
                    # Handle exit
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    # Find move based on player interaction
                    if event.type == pygame.MOUSEBUTTONDOWN and self.player1_turn:
                        pos = pygame.mouse.get_pos()
                        move = gui.pos_to_index(pos, self)
            else:  # AI player move (Player1)
                move = self.player1Search.strategy.find_move()

        else:  # AI player move (Player2)
            move = self.player2Search.strategy.find_move()

        # Update state based on the selected action
        try:
            # Check for illegal action
            assert self.is_move_possible(move), "Move has already been made"

            # Do the move
            self.make_move(move)

        except AssertionError as e:
            return False

        # print(self.player1Search.board)

        # Check if all ships are sunk
        self.check_game_over()

        # Change turn
        self.player1_turn = not self.player1_turn

    def is_move_possible(self, move):
        player = self.player1Search if self.player1_turn else self.player2Search

        if move is None:
            return False
        if player.board[0][move] != 0:
            return False
        return True

    def make_move(self, move):
        player = self.player1Search if self.player1_turn else self.player2Search
        opponent = self.player2Placement if self.player1_turn else self.player1Placement

        player.board[0][move] = 1  # Not unknown
        if move in opponent.indexes:
            player.board[1][move] = 1  # Hit
            self.check_ship_sunk(player, opponent, move)
        else:
            player.board[2][move] = 1  # Miss

    def check_ship_sunk(self, player, opponent, move):
        hit_ship = None
        sunk = True

        # Find the ship that was hit
        for ship in opponent.ships:
            if move in ship.indexes:
                hit_ship = ship.indexes

        # Check if the ship is sunk
        for i in hit_ship:
            if player.board[0][i] == 0:
                sunk = False
                break

        # If the ship is sunk, update the search board
        if sunk:
            for i in hit_ship:
                player.board[3][i] = 1

    def check_game_over(self):
        player = self.player1Search if self.player1_turn else self.player2Search
        opponent = self.player2Placement if self.player1_turn else self.player1Placement

        all_sunk = True
        for i in opponent.indexes:
            if player.board[0][i] == 0:
                all_sunk = False
        self.game_over = all_sunk
        self.result = 1 if self.player1_turn else 2
