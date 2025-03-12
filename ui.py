from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QGridLayout, QGraphicsTextItem, QGraphicsProxyWidget
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter
from PyQt6.QtCore import QSize, QTimer, Qt
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
        self.scene.setBackgroundBrush(QColor(25, 20, 20))
        self.setScene(self.scene)
        self.showMaximized()

        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Load the board image
        board_pixmap = QPixmap("./ui_assets/board_2.png")
        board_item = QGraphicsPixmapItem(board_pixmap)
        self.scene.addItem(board_item)

        self._width = board_pixmap.width()
        self._height = board_pixmap.height()

        self.tile_size = self._width // 7

        self.buttons = []  # Store buttons for later access
        self.setup_ui()

        # Load the numbers over the board
        numbers_pixmap = QPixmap("./ui_assets/numbers_2.png")
        numbers_item = QGraphicsPixmapItem(numbers_pixmap)
        numbers_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(numbers_item)

        padding = -50
        self.scene.setSceneRect(padding, padding, self._width - 2*padding, self._height - 2*padding)

        # Place pieces
        self.pieces = [[None]*7,[None]*7,[None]*7,[None]*7]
        self.place_pieces()
        self.selected_piece = None

        self.last_move_highlight = [(),(),(),()]

        self.valid_moves = self.board.get_valid_moves()

        self.use_ai = ai_path != ""
        self.ai_path = ai_path
        if self.use_ai:
            self.initialize_ai()

    def resizeEvent(self, event):
        """Resize the game board when the window resizes."""
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def initialize_ai(self):
        args = {
                "num_simulation": 1600,
                "c_base": 19652,
                "c_init": 1.25,
                "dirichlet_epsilon": 0,
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
        QTimer.singleShot(10, self.ai_move)


    def place_piece(self, path, pos_x, pos_y):
        piece_pixmap = QPixmap(path)
        piece_item = QGraphicsPixmapItem(piece_pixmap)
        piece_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
    
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
                    button.setFixedSize(QSize(self.tile_size-4, self.tile_size-4))  # Set size of each tile
                    button.setStyleSheet(
                        "QPushButton { background: transparent; border: none; }"
                        "QPushButton:hover { background: transparent; border: none; }"
                    )

                    button.clicked.connect(lambda _, r=row, c=col: self.on_click((r,c)))  # Connect click event
                    button.setGeometry(col * 180+4, row * 180+4, 180, 180)
                    self.scene.addWidget(button)
                    button_row.append(button)
                else:
                    button_row.append(None)
            self.buttons.append(button_row)

        self.setLayout(layout)

    def pos_to_tile(self, pos):
        return chr(pos[1] + 97) + str(4-pos[0])
    
    def tile_to_pos(self, tile):
        if len(tile) != 2:
            return (0,0)
        return (4-int(tile[1]), ord(tile[0]) - 97)
    
    def deselect_all_tiles(self):
        for r in range(4):
            for c in range(7):
                self.recolor_button((r,c), (0,0,0,0))

        self.recolor_button(self.last_move_highlight[0], (135, 140, 85, 180))
        self.recolor_button(self.last_move_highlight[1], (135, 140, 85, 180))
        self.recolor_button(self.last_move_highlight[2], (170, 180, 85, 180))
        self.recolor_button(self.last_move_highlight[3], (170, 180, 85, 180))

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
        game_over_text.setFont(QFont("Arial", 64, QFont.Weight.Bold))
        print(self._width, game_over_text.boundingRect().width())
        game_over_text.setPos((self._width-game_over_text.boundingRect().width())/2, (self._height-game_over_text.boundingRect().height())/2 - 50)  # Adjust position
        self.scene.addItem(game_over_text)
        
        # Create restart button
        restart_button = QPushButton("Restart")
        restart_button.setFixedSize(240, 100)
        restart_button.setStyleSheet("background-color: white; color: black; font-size: 32px;")

        # Connect restart function
        restart_button.clicked.connect(self.restart_game)

        # Place button on the scene
        proxy_widget = self.scene.addWidget(restart_button)
        proxy_widget.setPos((self._width-proxy_widget.boundingRect().width())/2, (self._height-proxy_widget.boundingRect().height())/2 + 50)  # Adjust position
            

    def make_move(self, move):
        print(move)
        self.board.move(move)
        start_pos = self.tile_to_pos(move[:2])
        end_pos = self.tile_to_pos(move[2:])
        self.move_piece(self.pieces[start_pos[0]][start_pos[1]], end_pos)
        if self.board.turn == 1:
            self.last_move_highlight[:2] = [start_pos, end_pos]
        else:
            self.last_move_highlight[2:] = [start_pos, end_pos]
        self.deselect_all_tiles()
        self.valid_moves = self.board.get_valid_moves()
        self.check_gameover()

    def ai_move(self):
        action, _, _ = self.A0.game_policy(self.board, random=False)
        self.make_move(action)
        #QTimer.singleShot(0, self.ai_move)

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

            for move in self.valid_moves:
                if move.startswith(tile):
                    self.recolor_button(self.tile_to_pos(move[2:]), (60,100,55,180))
        elif self.selected_piece is not None:
            move = self.pos_to_tile(self.selected_piece)+tile
            if move in self.valid_moves:
                self.make_move(move)
                if self.use_ai:
                    QTimer.singleShot(10, self.ai_move)

    def recolor_button(self, pos, color):
        """Highlight the button when clicked."""
        if pos == ():
            return
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
    ai_path = r'model_checkpoint_cycle_20.pth'
    # game = BoardGame(board)
    game = BoardGame(board, ai_path)
    game.show()
    sys.exit(app.exec())
