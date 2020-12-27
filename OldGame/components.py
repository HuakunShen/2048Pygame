import math
import numpy as np
import pandas as pd
from constants import *


class Tile:
    def __init__(self, power=None):
        self.power = power
        self.value = int(2 ** self.power) if self.power else None
        self.color = (255, 255, 255)
        self.bg_color = Tile.translate_bg_color(self.power)
        self.has_merged = False  # in one round, a tile can be merged only once

    @staticmethod
    def translate_bg_color(val):
        bg_color = None
        if type(val) is int and val > 11:
            bg_color = BG_COLOR_MAP[6]
        elif type(val) is int and val < 0:
            raise ValueError("Negative Number is invalid")
        elif val in BG_COLOR_MAP:
            bg_color = BG_COLOR_MAP[val]
        else:
            raise ValueError("Invalid Input")
        return bg_color

    def set_has_merged(self, value):
        self.has_merged = value

    def get_has_merged(self):
        return self.has_merged

    def increment(self):
        """
        increment tile power and value
        :return:
        """
        self.update_by_power(self.power + 1)

    def update_color(self):
        self.bg_color = Tile.get_color_by_power(self.power)

    @staticmethod
    def get_color_by_power(power):
        return Tile.translate_bg_color(power)

    @staticmethod
    def get_color_by_value(value):
        if type(value) == int:
            power = math.log(value, 2)
            if type(power) == float and power == power // 1:
                power = int(power // 1)
            else:
                raise ValueError("Error: value input is not correct")
        else:
            return Tile.translate_bg_color(value)
        return Tile.translate_bg_color(power)

    @staticmethod
    def power_to_value(power):
        if type(power) == int:
            return int(2 ** power)
        else:
            return None

    @staticmethod
    def value_to_power(value):
        if value is None:
            return None
        if type(value) == int:
            power = math.log(value, 2)
            if power != int(power):
                raise ValueError(
                    "value cannot be converted to power, must be power of 2")
        else:
            raise ValueError("value must be int")
        return int(power)

    def get_bg_color(self):
        return self.bg_color

    def get_color(self):
        return self.color

    def get_power(self):
        return self.power

    def update_by_power(self, power):
        self.set_power(power)
        self.value = Tile.power_to_value(power)
        self.update_color()

    def set_power(self, power):
        self.power = power

    def get_value(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Board:
    def __init__(self, dimension=4, matrix=None):
        # initialize grid to 4x4 tile board
        self.grid = None
        self.dimension = dimension
        self.bg_color = BG_COLOR_MAP["BACKGROUND"]
        if matrix is not None:
            self.construct_with_matrix(matrix)
        else:
            self.grid = [[Tile() for i in range(dimension)]
                         for j in range(dimension)]

    def construct_with_matrix(self, matrix):
        num_row = len(matrix)
        for row in matrix:
            if len(row) != num_row:
                raise Exception("not a matrix, must be nxn matrix")
            for val in row:
                if type(val) is not int and val is not None:
                    raise Exception(
                        "matrix value invalid, must be int or None")
        # construct
        self.grid = [[Tile(Tile.value_to_power(val))
                      for val in matrix_row] for matrix_row in matrix]

    def get_value_board(self):
        """
        convert self(board object) to a 2D array of pure integers (values)
        :return: 2D arrays of values
        """
        return np.array([[tile.get_value() for tile in row] for row in self.grid])

    def get_power_board(self):
        """
        convert self(board object) to a 2D array of pure integers (powers)
        :return: 2D arrays of powers
        """
        return np.array([[tile.get_power() for tile in row] for row in self.grid])

    def get_grid(self):
        return self.grid

    def get_tile(self, coordinate):
        return self.grid[coordinate[0]][coordinate[1]]

    def set_tile_power(self, coordinate, power):
        self.get_tile(coordinate).update_by_power(power)

    def rotate_cw(self):
        """
       rotate self.grid counterclockwise
       :return:
       """
        result = []
        for i in range(len(self.grid)):
            row = [r[i] for r in self.grid]
            row.reverse()
            result.append(row)
        self.grid = result

    def rotate_ccw(self):
        """
        rotate self.grid counterclockwise
        :return:
        """
        result = []
        for i in range(len(self.grid) - 1, -1, -1):
            row = [r[i] for r in self.grid]
            result.append(row)
        self.grid = result

    def get_empty_tiles_pos(self):
        result = []
        for row_index in range(len(self.grid)):
            row = self.grid[row_index]
            for col_index in range(len(row)):
                if row[col_index].get_power() is None:
                    result.append((row_index, col_index))
        return result

    def has_empty_tiles(self):
        return len(self.get_empty_tiles_pos()) == 0

    def move(self, direction):
        score = 0
        changed = False
        if direction == UP:
            self.rotate_cw()
            self.rotate_cw()
            score, changed = self.move_down()
            self.rotate_cw()
            self.rotate_cw()
        elif direction == DOWN:
            score, changed = self.move_down()
        elif direction == LEFT:
            self.rotate_ccw()
            score, changed = self.move_down()
            self.rotate_cw()
        elif direction == RIGHT:
            self.rotate_cw()
            score, changed = self.move_down()
            self.rotate_ccw()
        return score, changed

    def __str__(self):
        return str(pd.DataFrame(data=np.array(self.get_value_board())))

    def move_down(self):
        """
        move all tiles down, merge if available
        :return:
        """
        score = 0
        changed = False
        for row_index in range(self.dimension - 1, -1, -1):  # 从倒数第二行数到第0行
            row = self.grid[row_index]
            # iterate through every tile in the row
            for col_index in range(len(row)):
                tile = row[col_index]
                if tile.power is None:  # we don't care if a tile is None, do not move it
                    continue
                # 从倒数第二行开始向下merge，直到第一行
                for row_index_next in range(row_index + 1, self.dimension):
                    # iterate through every tile in the column
                    next_tile = self.grid[row_index_next][col_index]
                    if next_tile.power is None:  # if next tile is empty then just move it
                        next_tile.update_by_power(tile.get_power())
                        tile.update_by_power(None)
                        tile = next_tile
                        changed = True
                    elif tile.power == next_tile.power and not next_tile.get_has_merged():
                        # if 2 tile has equal power/value, merge them and delete one
                        next_tile.increment()
                        next_tile.set_has_merged(True)
                        tile.update_by_power(None)
                        score += next_tile.get_value()
                        changed = True
                        break
                    else:
                        # next tile and current tile have unequal value, and is not None, so curr tile stops where it is
                        break

        # clear has merged for all tiles
        for row in self.grid:
            for tile in row:
                tile.set_has_merged(False)
        return score, changed

    def get_available_neighbors(self, row, col):
        result = []
        if row > 0:
            result.append((row - 1, col))  # add left neighbor
        if row < self.dimension - 1:
            result.append((row + 1, col))  # add right neighbor
        if col > 0:
            result.append((row, col - 1))
        if col < self.dimension - 1:
            result.append((row, col + 1))
        return result


if __name__ == "__main__":
    m = [
        [2, 2, 4, 8],
        [2, 2, 4, 8],
        [2, 2, 4, 8],
        [2, 2, 4, 8],
    ]
    m = np.array([[2, None, None, None],
                  [2, None, None, None],
                  [None, None, None, None],
                  [None, None, None, None]])
    # board = Board()
    board = Board(matrix=m)
    print(board)
    print(board.get_value_board())
    print(board.get_power_board())
