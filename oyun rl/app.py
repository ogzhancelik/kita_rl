from env_final import Kita
import tkinter as tk
from tkinter import messagebox

class KitaGameUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kita Game")
        
        self.game = Kita()
        self.selected_piece = None
        
        self.buttons = [[None for _ in range(7)] for _ in range(4)]
        self.labels = [[None for _ in range(7)] for _ in range(4)]
        self.turn_label = tk.Label(self.root, text=f"Turn: {self.game.turn}")
        self.turn_label.grid(row=4, columnspan=7)
        self.step_label = tk.Label(self.root, text=f"Step size: {self.game.current_step_size}")
        self.step_label.grid(row=5, columnspan=7)
        
        self.create_board()
        self.update_board()

    def create_board(self):
        row_map = {0: '4', 1: '3', 2: '2', 3: '1'}
        col_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
        
        for r in range(4):
            for c in range(7):
                btn = tk.Button(self.root, text="", width=6, height=3, command=lambda row=r, col=c: self.on_click(row, col))
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn
                
                if self.game.tile_values[r][c] != 0:
                    label = tk.Label(self.root, text=self.game.tile_values[r][c], font=("Arial", 8))
                    label.config(fg="red")
                    label.place(x=c*83, y=r*90)
                    self.labels[r][c] = label

    def on_click(self, row, col):
        if self.selected_piece is None:
            if (row, col) in self.game.pieces[self.game.turn]:
                self.selected_piece = (row, col)
                self.buttons[row][col].config(bg="yellow")
        else:
            move = [self.selected_piece, (row, col)]
            if move in self.game.get_valid_moves():
                self.game.move(self.convert_move(move))
                self.update_board()
                self.selected_piece = None
                if self.game.check_gameover():
                    messagebox.showinfo("Game Over", "No valid moves left!")
                    self.game = Kita()
                    self.update_board()
            else:
                self.buttons[self.selected_piece[0]][self.selected_piece[1]].config(bg="SystemButtonFace")
                self.selected_piece = None
    
    def convert_move(self, move):
        row_map = {0: '4', 1: '3', 2: '2', 3: '1'}
        col_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
        return f"{col_map[move[0][1]]}{row_map[move[0][0]]}{col_map[move[1][1]]}{row_map[move[1][0]]}"
    
    def update_board(self):
        for r in range(4):
            for c in range(7):
                cell = self.game.board[r][c]
                text = "" if cell is None else str(cell)
                bg_color = "black" if cell is None else "SystemButtonFace"
                self.buttons[r][c].config(text="" if cell == 0 else text, bg=bg_color, fg="white" if cell is None else "black", font=("Arial", 16))
        self.turn_label.config(text=f"Turn: {self.game.turn}")
        self.step_label.config(text=f"Step size: {self.game.current_step_size}")

if __name__ == "__main__":
    root = tk.Tk()
    app = KitaGameUI(root)
    root.mainloop()
