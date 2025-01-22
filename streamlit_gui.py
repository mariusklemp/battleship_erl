import streamlit as st

from game_logic.game_manager import Game


class GUI:
    def __init__(self, board_size):
        self.board_size = board_size
        self.square_size = 40

    def draw_board(self, game):
        """Displays the game in Streamlit."""
        # Layout for boards with padding
        col1, spacer, col2 = st.columns([1, 0.1, 1], gap="medium")

        with col1:
            st.header("Player 1: Search")
            for row in range(self.board_size):
                cols = st.columns(self.board_size, gap="small")
                for col in range(self.board_size):
                    button_key = f"p1-search-{row}-{col}"  # Unique key for Player 1 search
                    if cols[col].button(" ", key=button_key, use_container_width=True):
                        index = row * self.board_size + col
                        game.play_turn(index)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Space between boards

            st.header("Player 1: Ships")
            for row in range(self.board_size):
                cols = st.columns(self.board_size, gap="small")
                for col in range(self.board_size):
                    button_key = f"p1-ship-{row}-{col}"  # Unique key for Player 1 ships
                    cols[col].button(" ", key=button_key, use_container_width=True)

        with col2:
            st.header("Player 2: Ships")
            for row in range(self.board_size):
                cols = st.columns(self.board_size, gap="small")
                for col in range(self.board_size):
                    button_key = f"p2-ship-{row}-{col}"  # Unique key for Player 2 ships
                    cols[col].button(" ", key=button_key, use_container_width=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Space between boards

            st.header("Player 2: Search")
            for row in range(self.board_size):
                cols = st.columns(self.board_size, gap="small")
                for col in range(self.board_size):
                    button_key = f"p2-search-{row}-{col}"  # Unique key for Player 2 search
                    if cols[col].button(" ", key=button_key, use_container_width=True):
                        index = row * self.board_size + col
                        game.play_turn(index)

        # Game over message
        if game.game_over:
            st.success(f"Game Over! Player {game.result} wins!")

        # Restart button
        if st.button("Restart Game"):
            game.reset()


# Example usage:
game = Game(board_size=10, sizes=[5, 4, 3, 3, 2], player1_search_strategy="random", player2_search_strategy="random",
            player1_placing_strategy="random", player2_placing_strategy="random")
gui = GUI(board_size=10)
gui.draw_board(game)
