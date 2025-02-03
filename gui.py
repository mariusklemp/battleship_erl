import pygame

pygame.init()
pygame.font.init()
pygame.display.set_caption("Battleship")
myfont = pygame.font.SysFont('fresansttf', 100)

GREY = (40, 50, 60)
WHITE = (255, 255, 255)
GREEN = (50, 200, 150)
RED = (250, 50, 100)
BLUE = (100, 150, 200)
ORANGE = (250, 140, 20)

COLORS_BOARD = {
    0: GREY,  # unknown
    1: ORANGE,  # hit
    2: BLUE,  # miss
    3: RED,  # sunk
}


class GUI:
    def __init__(self, board_size):
        self.BOARD_SIZE = board_size
        self.SQUARE_SIZE = 30
        self.H_MARGIN = self.SQUARE_SIZE * 4
        self.V_MARGIN = self.SQUARE_SIZE
        self.WIDTH = self.SQUARE_SIZE * self.BOARD_SIZE * 2 + self.H_MARGIN
        self.HEIGHT = self.SQUARE_SIZE * self.BOARD_SIZE * 2 + self.V_MARGIN
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.INDENT = 7

    def draw_grid(self, state, left=0, top=0, search=False):
        for i in range(self.BOARD_SIZE ** 2):
            x = left + i % self.BOARD_SIZE * self.SQUARE_SIZE
            y = top + i // self.BOARD_SIZE * self.SQUARE_SIZE
            square = pygame.Rect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
            pygame.draw.rect(self.SCREEN, WHITE, square, 1)

            if search:
                pygame.draw.circle(self.SCREEN, self.get_cell_color(state.board, i),
                                   (x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2),
                                   self.SQUARE_SIZE // 4)

    def get_cell_color(self, search, index):
        if search[3][index] == 1:  # sunk
            return RED
        elif search[1][index] == 1:  # hit
            return ORANGE
        elif search[2][index] == 1:  # miss
            return BLUE
        else:  # unknown
            return GREY

    def draw_ships(self, player, left=0, top=0):
        for ship in player.ships:
            x = left + ship.col * self.SQUARE_SIZE + self.INDENT
            y = top + ship.row * self.SQUARE_SIZE + self.INDENT

            if ship.direction == 0:  # horizontal
                width = ship.size * self.SQUARE_SIZE - 2 * self.INDENT
                height = self.SQUARE_SIZE - 2 * self.INDENT
            else:  # vertical
                width = self.SQUARE_SIZE - 2 * self.INDENT
                height = ship.size * self.SQUARE_SIZE - 2 * self.INDENT
            rectangle = pygame.Rect(x, y, width, height)
            pygame.draw.rect(self.SCREEN, GREEN, rectangle, border_radius=15)

    def pos_to_index(self, pos, game):
        x, y = pos
        index = None
        # player 1 clicks inside the board
        if not game.game_over and game.player1_turn and x < self.SQUARE_SIZE * self.BOARD_SIZE and y < self.SQUARE_SIZE * self.BOARD_SIZE:
            row = y // self.SQUARE_SIZE
            col = x // self.SQUARE_SIZE
            index = row * self.BOARD_SIZE + col

        # player 2 clicks inside the board
        elif not game.game_over and not game.player1_turn and x > self.WIDTH - self.SQUARE_SIZE * self.BOARD_SIZE and y > self.SQUARE_SIZE * self.BOARD_SIZE + self.V_MARGIN:
            row = (y - self.SQUARE_SIZE * self.BOARD_SIZE - self.V_MARGIN) // self.SQUARE_SIZE
            col = (x - self.SQUARE_SIZE * self.BOARD_SIZE - self.H_MARGIN) // self.SQUARE_SIZE
            index = row * self.BOARD_SIZE + col

        return index

    def update_board(self, state_1, state_2=None):
        """Updates the GUI board. Handles both two-player mode and single-agent mode."""

        # Draw background
        self.SCREEN.fill(GREY)

        # Draw search grid for Player 1
        self.draw_grid(state=state_1, search=True)

        # If playing against a placing agent (single-agent mode), don't draw Player 2's board
        if state_2 is not None:
            self.draw_grid(
                state=state_2, search=True,
                left=(self.WIDTH - self.H_MARGIN) // 2 + self.H_MARGIN,
                top=(self.HEIGHT - self.V_MARGIN) // 2 + self.V_MARGIN
            )

        # Draw position grid for Player 1
        self.draw_grid(state=state_1, top=(self.HEIGHT - self.V_MARGIN) // 2 + self.V_MARGIN)

        if state_2 is not None:
            self.draw_grid(
                state=state_2, left=(self.WIDTH - self.H_MARGIN) // 2 + self.H_MARGIN
            )

        # Draw Player 1's ships
        self.draw_ships(state_1.placing, top=(self.HEIGHT - self.V_MARGIN) // 2 + self.V_MARGIN)

        # Draw Player 2's ships only if it exists
        if state_2 is not None:
            self.draw_ships(
                state_2.placing, left=(self.WIDTH - self.H_MARGIN) // 2 + self.H_MARGIN
            )

    def display_win(self, result, move_count):
        """Displays the game-over message clearly and centered."""

        # Create a new font with a more adaptive size
        font_size = max(40, self.SQUARE_SIZE * 2)  # Adjust based on board size
        myfont = pygame.font.SysFont('fresansttf', font_size)

        # Create text surfaces
        text1 = f"Player {result} wins!"
        text2 = f"{move_count}/{self.BOARD_SIZE * self.BOARD_SIZE} moves"

        text1_surface = myfont.render(text1, True, WHITE)
        text2_surface = myfont.render(text2, True, WHITE)

        # Get text positions to center dynamically
        text1_rect = text1_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 30))
        text2_rect = text2_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 30))  # Slightly below text1

        # Fill the background with a slightly transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Dark transparent background
        self.SCREEN.blit(overlay, (0, 0))

        # Draw the text on screen
        self.SCREEN.blit(text1_surface, text1_rect)
        self.SCREEN.blit(text2_surface, text2_rect)

        # Update display
        pygame.display.flip()

        # Pause for a few seconds or wait for user input before closing
        pygame.time.wait(2000)  # 2-second delay


