import sys
import copy
import random
import constants
import numpy as np
from components import Board, Tile


class GameState:
    def __init__(self, dimension=4, score=0, matrix=None):
        self._score = score
        self._dimension = dimension
        self._init_matrix = matrix
        self._board = Board(matrix=self._init_matrix)
        random.seed(constants.RANDOM_SEED)
        if len(self._board.get_empty_tiles_pos()) != 0 and matrix is None:
            for _ in range(2):
                self.set_random_tile()
        self.is_done = self.compute_is_done()

    def get_is_done(self):
        return self.is_done

    def get_board(self):
        return self._board

    def get_dimension(self):
        return self._dimension

    def get_score(self):
        return self._score

    def compute_is_done(self):
        value_matrix = self._board.get_value_board()
        for row_index in range(len(self._board.grid)):
            row = self._board.grid[row_index]
            for col_index in range(len(row)):
                neighbors = self._board.get_available_neighbors(
                    row_index, col_index)
                for neighbor in neighbors:
                    if value_matrix[neighbor[0]][neighbor[1]] == value_matrix[row_index][col_index]:
                        return False
                    if value_matrix[row_index][col_index] is None:
                        return False
        return True

    def _get_new_tile_position(self):
        empty_pos = self._board.get_empty_tiles_pos()
        target_pos = empty_pos[random.randint(0, len(empty_pos) - 1)]
        return target_pos

    def set_random_tile(self):
        is_4 = random.random() < 0.1
        target_pos = self._get_new_tile_position()
        self._board.set_tile_power(target_pos, 2 if is_4 else 1)

    def restart(self):
        random.seed(constants.RANDOM_SEED)
        self.move(constants.K_r)

    def move(self, key):
        score = 0
        changed = False
        if key == constants.K_r:
            print('restart')
            self.__init__(matrix=self._init_matrix)
        elif key == constants.K_q:
            print('quit game')
            sys.exit(0)
        else:
            if not self.is_done:
                score, changed = self._board.move(key)
                self.is_done = self.compute_is_done()
            self._score += score
            if changed:
                # add a random tile, 2 or 4
                self.set_random_tile()
        return score, changed

    def random_move(self):
        return self.move(constants.ARROW_KEYS[random.randint(0, 3)])

    def copy(self):
        return copy.deepcopy(self)


if __name__ == '__main__':
    m = np.array([[2, None, None, 8],
                  [2, None, None, 8],
                  [None, None, None, None],
                  [None, None, None, None]])
    game = GameState(matrix=m)
    print(game.get_board())
    game.move(constants.DOWN)
    print(game.get_board())
