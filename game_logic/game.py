import sys
import pygame


class Game:
    """
    Class to manage the game state between two players:
    - Search 1 plays against Placing 2
    - Search 2 plays against Placing 1
    """

    def __init__(self, game_manager1, game_manager2, player1_search, player2_search):
        self.player1_turn = True  # Player 1 starts
        self.game_manager1 = game_manager1  # Manages Player 1's board
        self.game_manager2 = game_manager2  # Manages Player 2's board
        self.player1_search = player1_search  # AI for Player 1
        self.player2_search = player2_search  # AI for Player 2
        self.game_over = False
        self.winner = None

    def play_turn(self, current_state_1, current_state_2, gui=None):
        move = None

        if self.player1_turn:  # Player 1 (Search 1) attacks Player 2’s board
            if gui:
                move = self.get_human_move(gui)  # Human move if GUI enabled
            else:
                move = self.player1_search.strategy.find_move(current_state_1)

        else:  # Player 2 (Search 2) attacks Player 1’s board
            move = self.player2_search.strategy.find_move(current_state_2)

        try:
            # Check for illegal action
            assert self.is_move_possible(move), "Move has already been made"

            if self.player1_turn:
                print(f"Player 1 trying to hit")
                self.game_manager1.placing.show_ships()
                print(f"Player 1 move: {move}")
                print("On the board:", current_state_1.board[0])
                current_state_1 = self.game_manager1.next_state(current_state_1, move)
                print("After move:", current_state_1.board[0])
            else:
                print(f"Player 2 trying to hit")
                self.game_manager2.placing.show_ships()
                print(f"Player 2 move: {move}")
                print("On the board:", current_state_2.board[0])
                current_state_2 = self.game_manager2.next_state(
                    current_state_2, move
                )
                print("After move:", current_state_2.board[0])

        except AssertionError as e:
            return current_state_1, current_state_2

        # Check if the game is over
        self.check_game_over(current_state_1, current_state_2)

        # Switch turn
        self.player1_turn = not self.player1_turn

        return current_state_1, current_state_2

    def get_human_move(self, gui):
        """Handles player move if using a GUI."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and self.player1_turn:
                pos = pygame.mouse.get_pos()
                return gui.pos_to_index(pos, self)
        return None

    def is_move_possible(self, move):
        """Checks if a move is legal."""
        if move is None:
            return False

        if self.player1_turn:
            game = self.game_manager2  # Player 1 is attacking Player 2’s board
        else:
            game = self.game_manager1  # Player 2 is attacking Player 1’s board

        return game.board[0][move] == 0  # Ensure the move has not been played

    def check_game_over(self, state1, state2):
        """Checks if a player has won the game."""
        if self.game_manager1.is_terminal(state1):
            print("Player 2 wins")
            self.game_over = True
            self.winner = "Player 2"  # Player 2 wins because Player 1 lost

        if self.game_manager2.is_terminal(state2):
            print("Player 1 wins")
            self.game_over = True
            self.winner = "Player 1"  # Player 1 wins because Player 2 lost
