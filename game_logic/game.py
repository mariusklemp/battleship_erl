import sys
import pygame


class Game:
    """
    Class to manage the game state between two players:
    - Search vs Search (standard Battleship)
    - Search vs Placing (single-agent mode)
    """

    def __init__(self, game_manager1, player1_search, game_manager2=None, player2_search=None):
        """
        Initialize the game.

        :param game_manager1: Manages Player 1's board
        :param game_manager2: Manages Player 2's board
        :param player1_search: AI search agent for Player 1
        :param player2_search: AI search agent for Player 2 (optional for single-agent mode)
        """
        self.game_manager1 = game_manager1  # Manages Player 1's board
        self.game_manager2 = game_manager2  # Manages Player 2's board
        self.player1_search = player1_search  # AI for Player 1
        self.player2_search = player2_search  # AI for Player 2 (optional)
        self.player1_turn = True  # Player 1 starts

        # Single-agent mode: No Player 2 search agent
        self.single_agent_mode = player2_search is None
        self.game_over = False
        self.winner = None

    def play_turn(self, current_state_1, current_state_2=None, gui=None):
        """
        Handles a turn in the game.

        - If single-agent mode is enabled, only Player 1 (search agent) makes moves.
        - Otherwise, turns alternate normally.

        :param current_state_1: Current state of Player 1's board
        :param current_state_2: Current state of Player 2's board
        :param gui: GUI instance (optional)
        :return: Updated board states
        """
        move = None

        if self.single_agent_mode:  # Only Player 1 plays (search vs placing)
            move = self.get_human_move(gui) if gui else self.player1_search.strategy.find_move(current_state_1)
        else:  # Regular Battleship (Search vs Search)
            if self.player1_turn:
                move = self.get_human_move(gui) if gui else self.player1_search.strategy.find_move(current_state_1)
            else:
                move = self.player2_search.strategy.find_move(current_state_2)

        try:
            # Validate move
            assert self.is_move_possible(move), "Move has already been made"

            if self.single_agent_mode or self.player1_turn:
                current_state_1 = self.game_manager1.next_state(current_state_1, move)
            else:
                current_state_2 = self.game_manager2.next_state(current_state_2, move)

        except AssertionError:
            return current_state_1, current_state_2  # Return unchanged if move is invalid

        # Check if the game is over
        self.check_game_over(current_state_1, current_state_2)

        # In regular mode, switch turns
        if not self.single_agent_mode:
            self.player1_turn = not self.player1_turn

        return current_state_1, current_state_2

    def get_human_move(self, gui):
        """Handles human move input using the GUI."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and self.player1_turn:
                pos = pygame.mouse.get_pos()
                return gui.pos_to_index(pos, self)
        return None


    def is_move_possible(self, move):
        """Checks if a move is valid (i.e., not already played)."""
        if move is None:
            return False

        # Single-agent mode: Only check Player 1's board
        if self.single_agent_mode:
            return self.game_manager1.board[0][move] == 0

        # Multiplayer mode: Check the appropriate player's board
        game = self.game_manager2 if self.player1_turn else self.game_manager1
        return game.board[0][move] == 0  # Move is valid if the board is empty at that position

    def check_game_over(self, state1, state2):
        """Checks if the game is over."""
        if self.game_manager1.is_terminal(state1):
            self.game_over = True
            self.winner = "Player 1" if not self.single_agent_mode else f'{self.player1_search.strategy.name}'

        if not self.single_agent_mode:
            if self.game_manager2.is_terminal(state2):
                self.game_over = True
                self.winner = f'{self.player2_search.strategy.name}'

