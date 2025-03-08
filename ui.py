from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QGridLayout, QGraphicsTextItem, QGraphicsProxyWidget
from PyQt6.QtGui import QPixmap, QFont, QColor 
from PyQt6.QtCore import QSize, QTimer
import sys
from env import Kita
from main import AlphaZero
import torch
import time

class BoardGame(QGraphicsView):
    def __init__(self, board:Kita, ai_path=""):
        super().__init__()

        self.board = board
        
        # Set up the scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Load the board image
        self.board_pixmap = QPixmap("./ui_assets/board.png")
        self.board_item = QGraphicsPixmapItem(self.board_pixmap)
        self.scene.addItem(self.board_item)

        # Define board size
        self.board_width = 7
        self.board_height = 4
        self.tile_size = self.board_pixmap.width() // self.board_width

        self.buttons = []  # Store buttons for later access
        self.setup_ui()

        # Place pieces
        self.pieces = [[None]*7,[None]*7,[None]*7,[None]*7]
        self.place_pieces()
        self.selected_piece = None

        self.valid_moves = self.board.get_valid_moves()

        self.use_ai = ai_path != ""
        self.ai_path = ai_path
        if self.use_ai:
            self.initialize_ai()

    def initialize_ai(self):
        args = {
                "num_simulation": 400,
                "c_base": 19652,
                "c_init": 1.25,
                "dirichlet_epsilon": 0.25,
                "dirichlet_alpha": 3.0,
                "action_space": 672,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "top_actions": 15,
                "t": 1,
                "step": 3,
                "model_checkpoint_path": "model_epoch_1.pth",
                "num_games": 1,  # Paralel olarak çalıştırılacak oyun sayısı
                "games_per_training": 96,  # Kaç oyun oynandıktan sonra eğitime geçileceği
                "max_cycles": 1000,  # Kaç defa 1000 oyun + eğitim döngüsü yapılacağı
            }
        self.A0 = AlphaZero(args)
        self.A0.load_model(self.ai_path)
        self.ai_move()


    def place_piece(self, path, pos_x, pos_y):
        piece_pixmap = QPixmap(path)
        piece_item = QGraphicsPixmapItem(piece_pixmap)
    
        self.move_piece(piece_item, (pos_y, pos_x))
        self.scene.addItem(piece_item)

    def move_piece(self, piece, pos):
        piece_width = piece.pixmap().width()
        offset = (self.tile_size - piece_width) / 2

        r,c = pos
        piece.setPos(c * self.tile_size + offset, r * self.tile_size + offset)

        self.pieces[r][c] = piece


    def place_pieces(self):
        self.place_piece("./ui_assets/black_kita.png", 0, 0)
        self.place_piece("./ui_assets/white_kita.png", 6, 3)
        self.place_piece("./ui_assets/black_piece.png", 0, 1)
        self.place_piece("./ui_assets/black_piece.png", 1, 0)
        self.place_piece("./ui_assets/white_piece.png", 5, 3)
        self.place_piece("./ui_assets/white_piece.png", 6, 2)

    def setup_ui(self):
        layout = QGridLayout()
        layout.setSpacing(0)  # Adjust spacing between buttons

        for row in range(4):
            button_row = []
            for col in range(7):
                if self.board.board[row][col] is not None:
                    button = QPushButton()
                    button.setFixedSize(QSize(90, 90))  # Set size of each tile
                    button.setStyleSheet(
                        "QPushButton { background: transparent; border: none; }"
                        "QPushButton:hover { background: transparent; border: none; }"
                    )

                    button.clicked.connect(lambda _, r=row, c=col: self.on_click((r,c)))  # Connect click event
                    button.setGeometry(col * 90, row * 90, 90, 90)
                    self.scene.addWidget(button)
                    button_row.append(button)
                else:
                    button_row.append(None)
            self.buttons.append(button_row)

        self.setLayout(layout)

    def pos_to_tile(self, pos):
        return chr(pos[1] + 97) + str(4-pos[0])
    
    def tile_to_pos(self, tile):
        return (4-int(tile[1]), ord(tile[0]) - 97)
    
    def deselect_all_tiles(self):
        for r in range(4):
            for c in range(7):
                self.recolor_button((r,c), (0,0,0,0))

    def restart_game(self):
        # Close the current window
        self.close()  # Close the current window

        # Restart the game by creating a new instance
        board = Kita()
        new_game = BoardGame(board, self.ai_path)  # Create a new game instance
        new_game.show()  # Show the new game window

    def check_gameover(self):
        if not self.board.check_gameover():
            return
    
        # Create 'Game Over' text
        game_over_text = QGraphicsTextItem("Game Over")
        game_over_text.setDefaultTextColor(QColor(255,0,0))
        game_over_text.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        game_over_text.setPos(200, 150)  # Adjust position
        self.scene.addItem(game_over_text)
        
        # Create restart button
        restart_button = QPushButton("Restart")
        restart_button.setFixedSize(120, 50)
        restart_button.setStyleSheet("background-color: white; color: black; font-size: 16px;")

        # Connect restart function
        restart_button.clicked.connect(self.restart_game)

        # Place button on the scene
        proxy_widget = self.scene.addWidget(restart_button)
        proxy_widget.setPos(250, 220)  # Adjust position
            

    def make_move(self, move):
        self.board.move(move)
        start_pos = self.tile_to_pos(move[:2])
        end_pos = self.tile_to_pos(move[2:])
        self.move_piece(self.pieces[start_pos[0]][start_pos[1]], end_pos)
        self.deselect_all_tiles()
        self.valid_moves = self.board.get_valid_moves()
        self.check_gameover()

    def ai_move(self):
        action, _, _ = self.A0.game_policy(self.board, random=False)
        self.make_move(action)

    def on_click(self, pos):
        if self.board.turn == 1 and self.use_ai:
            return
        tile = self.pos_to_tile(pos)
        if self.board.board[pos[0]][pos[1]] in [1*self.board.turn,2*self.board.turn]:
            if self.selected_piece is not None:
                self.deselect_all_tiles()
            if self.selected_piece == pos:
                self.selected_piece = None
                return
            self.selected_piece = pos
            self.recolor_button(pos, (60,100,55,180))

            
        elif self.selected_piece is not None:
            move = self.pos_to_tile(self.selected_piece)+tile
            if move in self.valid_moves:
                self.make_move(move)
                if self.use_ai:
                    QTimer.singleShot(10, self.ai_move)

    def recolor_button(self, pos, color):
        """Highlight the button when clicked."""
        button = self.buttons[pos[0]][pos[1]]
        if button is None:
            return
        button.setStyleSheet(
            f"QPushButton {{ background: rgba{color}; border: none; }} "
            f"QPushButton:hover {{ background: rgba{color}; }} "
        )



if __name__ == "__main__":
    board = Kita()
    app = QApplication(sys.argv)
    game = BoardGame(board, r'model_checkpoint_cycle_3.pth')
    game.show()
    sys.exit(app.exec())
