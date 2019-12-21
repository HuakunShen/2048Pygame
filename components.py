import math
import numpy as np
from constants import *


class Tile:
    bg_colors = {
        "BACKGROUND": (172, 157, 142),
        None: (193, 179, 165),
        1: (233, 221, 209),
        2: (232, 217, 187),
        3: (235, 160, 99),
        4: (237, 128, 78),
        5: (237, 100, 75),
        6: (236, 69, 43),
        7: (231, 197, 90),
        8: (231, 197, 91),
        9: (231, 188, 55),
        10: (230, 185, 38),
        11: (230, 181, 19)
    }

    def __init__(self, power=None):
        self.power = power
        self.value = int(2 ** self.power) if self.power else None
        self.color = (255, 255, 255)
        self.bg_color = Tile.bg_colors[self.power]
        self.has_merged = False  # in one round, a tile can be merged only once

    def set_has_merged(self, value):
        self.has_merged = value

    def get_has_merged(self):
        return self.has_merged

    def increment(self):
        """
        increment tile power and value
        :return:
        """
        self.power += 1
        self.value = int(2 ** self.power) if self.power else None
        self.update_color()

    def update_color(self):
        self.bg_color = Tile.get_color_by_power(self.power)

    @staticmethod
    def get_color_by_power(power):
        if power in Tile.bg_colors:
            return Tile.bg_colors[power]
        else:
            return Tile.bg_colors[6]

    @staticmethod
    def get_color_by_value(value):
        if type(value) == int:
            power = math.log(value, 2)
            if type(power) == float and power == power // 1:
                power = power // 1
            else:
                print("Error: value input is not correct")
                exit(1)
        else:
            power = 6
        return Tile.bg_colors[power]

    @staticmethod
    def power_to_value(power):
        if type(power) == int:
            return int(2 ** power)
        else:
            return None

    @staticmethod
    def value_to_power(value):
        if type(value) == int:
            power = math.log(value, 2)
            if power != int(power):
                raise Exception("value cannot be converted to power, must be power of 2")
        else:
            raise Exception("value must be int")
        return int(power)

    def get_bg_color(self):
        return self.bg_color

    def get_color(self):
        return self.color

    def get_power(self):
        return self.power

    def set_power(self, power):
        self.power = power
        self.value = Tile.power_to_value(power)
        self.update_color()

    def get_value(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Board:

    def __init__(self, dimension=4, matrix=None):
        # initialize grid to 4x4 tile board
        self.grid = []
        self.dimension = dimension
        self.bg_color = (172, 157, 142)
        if matrix:
            self.construct_with_matrix(matrix)
            return
        for i in range(4):
            row = []
            for j in range(4):
                row.append(Tile())
            self.grid.append(row)

    def construct_with_matrix(self, matrix):
        num_row = len(matrix)
        for row in matrix:
            if len(row) != num_row:
                raise Exception("not a matrix, must be nxn matrix")
            for val in row:
                if type(val) is not int:
                    raise Exception("matrix value invalid, must be all int")
        # construct
        self.grid = []
        for matrix_row in matrix:
            row = []
            for val in matrix_row:
                row.append(Tile(Tile.value_to_power(val)))
            self.grid.append(row)

    def get_value_board(self):
        """
        convert self(board object) to a 2D array of pure integers (values)
        :return: 2D arrays of values
        """
        value_board = []
        for row in self.grid:
            value_row = [tile.get_value() for tile in row]
            value_board.append(value_row)
        return value_board

    def get_power_board(self):
        """
        convert self(board object) to a 2D array of pure integers (powers)
        :return: 2D arrays of powers
        """
        power_board = []
        for row in self.grid:
            power_row = [tile.get_power() for tile in row]
            power_board.append(power_row)
        return power_board

    def get_grid(self):
        return self.grid

    def get_tile(self, coordinate):
        return self.grid[coordinate[0]][coordinate[1]]

    def set_tile_power(self, coordinate, power):
        self.get_tile(coordinate).set_power(power)

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
                col = row[col_index]
                if self.grid[row_index][col_index].get_power() is None:
                    result.append((row_index, col_index))
        return result

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
        result = ''
        for row in self.grid:
            for tile in row:
                value = str(0) if not tile.get_value() else str(tile.get_value())
                result += value + ', '
            result += '\n'
        return result

    def move_down(self):
        """
        move all tiles down, merge if available
        :return:
        """
        score = 0
        changed = False
        for row_index in range(self.dimension - 1, -1, -1):  # 从倒数第二行数到第0行
            row = self.grid[row_index]
            for col_index in range(len(row)):  # iterate through every tile in the row
                tile = row[col_index]
                if tile.power is None:  # we don't care if a tile is None, do not move it
                    continue
                for row_index_next in range(row_index + 1, self.dimension):  # 从倒数第二行开始向下merge，直到最后一行
                    # iterate through every tile in the column
                    next_tile = self.grid[row_index_next][col_index]
                    if next_tile.power is None:  # if next tile is empty then just move it
                        next_tile.set_power(tile.get_power())
                        tile.set_power(None)
                        tile = next_tile
                        changed = True
                    elif tile.power == next_tile.power and not next_tile.get_has_merged():
                        # if 2 tile has equal power/value, merge them and delete one
                        next_tile.increment()
                        next_tile.set_has_merged(True)
                        tile.set_power(None)
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

    def is_done(self):
        value_matrix = self.get_value_board()
        for row_index in range(len(self.grid)):
            row = self.grid[row_index]
            for col_index in range(len(row)):
                neighbors = self.get_available_neighbors(row_index, col_index)
                for neighbor in neighbors:
                    if value_matrix[neighbor[0]][neighbor[1]] == value_matrix[row_index][col_index]:
                        return False
                    if value_matrix[row_index][col_index] is None:
                        return False
        return True

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
