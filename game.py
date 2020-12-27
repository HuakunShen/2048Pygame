import sys
import copy
import torch
import random
import constants
import pygame
from numba import jit, njit
from constants import UP, DOWN, LEFT, RIGHT, K_r, K_q
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from torch import Tensor


class Board(object):
    def __init__(self, grid_w: int = 4, grid_h: int = 4, init_matrix: Union[None, np.ndarray] = None, seed=constants.SEED):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.init_grid(init_matrix)

    def init_grid(self, init_matrix: Union[None, np.ndarray]):
        self.grid = np.zeros((self.grid_w, self.grid_h)
                             ) if self.init_matrix is None else self.init_matrix

    def get_grid(self):
        return self.grid

    def get_pd_grid(self):
        return pd.DataFrame(data=self.grid)

    def __str__(self):
        return str(self.get_pd_grid())

    def get_empty_coordinates(self):
        return np.argwhere(self.grid == 0)

    def has_empty_cell(self):
        return len(self.get_empty_coordinates()) != 0


def get_bg_color(val: Union[int, str]):
    bg_color = None
    if type(val) is int and val > 2048:
        bg_color = constants.CELL_BG_COLOR_MAP[64]
    elif type(val) is int and val < 0:
        raise ValueError("Negative Number is invalid")
    elif val in constants.CELL_BG_COLOR_MAP:
        bg_color = constants.CELL_BG_COLOR_MAP[val]
    else:
        raise ValueError(f"Invalid Input: {val}")
    return bg_color


class TorchStaticBoard(object):
    seed = 2048

    @staticmethod
    def get_pd_df(matrix: Tensor) -> pd.DataFrame:
        return pd.DataFrame(data=matrix).astype(int)

    @staticmethod
    def get_empty_coordinates(matrix: Tensor) -> Tensor:
        return torch.nonzero(matrix == 0)

    @staticmethod
    def has_empty_cell(matrix: Tensor) -> bool:
        return len(TorchStaticBoard.get_empty_coordinates(matrix)) != 0

    @staticmethod
    def has_won(matrix: Tensor, goal: int = 2048):
        return goal in matrix

    def get_max_val(matrix: Tensor):
        return int(torch.max(matrix))

    @staticmethod
    def get_random_empty_cell_coordinate(matrix: Tensor) -> Union[Tensor, None]:
        empty_cell_coordinates = TorchStaticBoard.get_empty_coordinates(matrix)
        return None if len(empty_cell_coordinates) == 0 else empty_cell_coordinates[random.randint(0, len(empty_cell_coordinates) - 1)]

    @staticmethod
    def get_empty_matrix(width: int = 4, height: int = 4):
        return torch.zeros((width, height)).int()

    @staticmethod
    def get_init_matrix(width: int = 4, height: int = 4):
        matrix = TorchStaticBoard.get_empty_matrix(width=width, height=height)
        for _ in range(2):
            TorchStaticBoard.set_random_cell(matrix, inplace=True)
        return matrix

    @staticmethod
    def set_random_cell(matrix: Tensor, inplace: bool = True) -> Tuple[Tensor, bool]:
        empty_cell = TorchStaticBoard.get_random_empty_cell_coordinate(matrix)
        if not inplace:
            matrix = matrix.clone()
        if empty_cell is None:
            return matrix, False
        else:
            is_4 = random.random() < 0.1
            matrix[empty_cell[0], empty_cell[1]] = 4 if is_4 else 2
            return matrix, True

    @staticmethod
    def get_neighbors_coordinates(matrix: Tensor, row_i: int, col_i: int) -> List:
        result = []
        h, w = matrix.shape
        if row_i > 0:
            result.append((row_i - 1, col_i))
        if row_i < h - 1:
            result.append((row_i + 1, col_i))
        if col_i > 0:
            result.append((row_i, col_i - 1))
        if col_i < w - 1:
            result.append((row_i, col_i + 1))
        return result

    @staticmethod
    def compute_is_done(matrix: Tensor):
        h, w = matrix.shape
        for row_i in range(h):
            row = matrix[row_i, :]
            for col_i in range(len(row)):
                neighbor_coordinates = TorchStaticBoard.get_neighbors_coordinates(
                    matrix, row_i, col_i)
                for neighbor in neighbor_coordinates:
                    if matrix[neighbor[0], neighbor[1]] == matrix[row_i, col_i]:
                        return False
                    if matrix[row_i, col_i] == 0:
                        return False
        return True

    @staticmethod
    def move(matrix: Tensor, direction: Union[UP, DOWN, LEFT, RIGHT], inplace: bool = True) -> Tuple[Tensor, bool]:
        has_merged_matrix = torch.zeros_like(matrix)
        score = 0
        changed = False
        num_row, num_col = matrix.shape
        if not inplace:
            matrix = matrix.clone()
        for i in range(len(matrix)):
            if direction == UP:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[:, i], reverse=True)
            elif direction == DOWN:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[:, i], reverse=False)
            elif direction == LEFT:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[i, :], reverse=True)
            elif direction == RIGHT:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[i, :], reverse=False)
            else:
                raise ValueError(f"Invalid direction: {direction}")
            changed = max(changed, changed_)
            score += score_
        return matrix, score, changed

    @staticmethod
    def collapse_array(arr: Tensor, reverse=False):
        """
        collapse an array in left-to-right direction in 2048-way inplace, this is either a row or an column
        :param array: Tensor Array
        :return:
        """
        arr_len = len(arr)
        changed = False
        score = 0
        has_merged_arr = torch.zeros_like(arr)
        for i in range(1 if reverse else arr_len - 2, arr_len if reverse else -1, 1 if reverse else -1):
            curr_i = i
            for next_i in range(i + (-1 if reverse else 1), -1 if reverse else arr_len, -1 if reverse else 1):
                if arr[next_i] == 0 and arr[curr_i] != 0:
                    arr[next_i] = arr[curr_i]
                    arr[curr_i] = 0
                    curr_i = next_i
                    changed = True
                elif arr[curr_i] == arr[next_i] and not has_merged_arr[next_i] and arr[curr_i] != 0:
                    # merge case
                    arr[next_i] *= 2
                    arr[curr_i] = 0
                    has_merged_arr[next_i] = 1
                    score += int(arr[next_i])
                    changed = True
                    break
                else:
                    break
        return arr, score, changed


class Game(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, width: int = 4, height: int = 4, seed: int = constants.SEED, goal: int = constants.DEFAULT_GOAL):
        self.goal = goal
        self.seed = seed
        random.seed(self.seed)
        self.matrix = matrix if matrix is not None else TorchStaticBoard.get_init_matrix(
            width, height)
        self.score = 0
        self.is_done = False
        self.width = width
        self.height = height

    def get_matrix(self):
        return self.matrix

    def get_score(self):
        return self.score

    def get_is_done(self):
        return self.is_done

    def has_won(self):
        return TorchStaticBoard.has_won(self.matrix, self.goal)

    def get_max_val(self):
        return TorchStaticBoard.get_max_val(self.matrix)

    def restart(self):
        self.__init__(matrix=None, width=self.width,
                      height=self.height, seed=self.seed)

    def move(self, direction: Union[UP, DOWN, LEFT, RIGHT, K_r, K_q, None] = None, inplace: bool = True):
        if direction == K_r:
            self.restart()
            return self.matrix, 0, True
        else:
            if direction is None:
                direction = constants.ARROW_KEYS[random.randint(0, 3)]
            matrix, score, changed = TorchStaticBoard.move(
                matrix=self.matrix, direction=direction, inplace=inplace)
            self.score += score
            self.is_done = TorchStaticBoard.compute_is_done(matrix)
            if changed:
                matrix, added = TorchStaticBoard.set_random_cell(
                    self.matrix, inplace=inplace)
            return matrix, score, changed

    def clone(self):
        return copy.deepcopy(self)


class GameUI(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, game: Game = None, width: int = 800, height: int = 950, margin: int = 10):
        self.game = Game(matrix=matrix) if game is None else game
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.margin = margin
        self.block_size = (
            self.width - (self.game.get_matrix().shape[0] + 1) * margin) // self.game.get_matrix().shape[0]
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048')

    def main(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in constants.KEY_MAP:
                        if constants.KEY_MAP[event.key] == constants.K_q:
                            pygame.quit()
                            print("quit 2048")
                            sys.exit(0)
                        else:
                            matrix, score, changed = self.game.move(
                                direction=constants.KEY_MAP[event.key], inplace=True)
            self.update_ui()

    def set_game(self, game):
        self.game = game

    def update_score(self):
        font = pygame.font.Font(None, 64)
        text = font.render(
            'Score: ' + str(self.game.get_score()), 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def update_msg(self):
        font = pygame.font.Font(None, 32)
        text = font.render('Game ends, press r to restart' if self.game.get_is_done(
        ) else "Click 'q' to quit the game", True, (255, 255, 255))
        self.screen.blit(text, (50, 870))

    def draw_grid(self):
        font = pygame.font.Font(None, 64)
        matrix = self.game.get_matrix()
        num_row, num_col = matrix.shape
        for row_i in range(num_row):
            row = matrix[row_i, :]
            for col_i in range(num_col):
                cell_val = int(matrix[row_i, col_i])
                rect = pygame.Rect(col_i * self.block_size + (col_i + 1) * self.margin, row_i * self.block_size + (row_i + 1) *
                                   self.margin,
                                   self.block_size,
                                   self.block_size)
                pygame.draw.rect(self.screen, get_bg_color(cell_val), rect)
                text_content = "" if cell_val == 0 else str(cell_val)
                text = font.render(text_content, True, (255, 255, 255))
                text_position = text.get_rect()
                text_position.center = rect.center
                self.screen.blit(text, text_position)

    def update_ui(self):
        self.clock.tick(30)
        self.screen.fill(constants.BG_COLOR)
        self.draw_grid()
        self.update_score()
        self.update_msg()
        pygame.display.flip()


if __name__ == "__main__":
    # matrix = TorchStaticBoard.get_init_matrix()
    # print(TorchStaticBoard.get_neighbors_coordinates(matrix, 1, 1))

    # game = Game()
    # print(game.get_matrix())
    # game_cpy = game.clone()
    # game_cpy.matrix[0, 0] = 2048
    # print(game_cpy.get_matrix())
    # print(game.get_matrix())

    game = GameUI()
    game.main()

    # t_matrix = torch.zeros((4, 4))
    # print(StaticBoard.get_pd_df(t_matrix))
    # print(StaticBoard.get_empty_coordinates(t_matrix))
    # print(StaticBoard.get_random_empty_cell_coordinate(t_matrix))
    # print(StaticBoard.set_random_cell(t_matrix, inplace=True))
    # print(t_matrix)
    # print(StaticBoard.set_random_cell(t_matrix, inplace=False))
    # print(t_matrix)

    # t_matrix = Tensor([[0., 0., 2., 2.],
    #                    [0., 2., 2., 0.],
    #                    [2., 2., 0., 0.],
    #                    [2., 0., 2., 2.]])

    # t_matrix = torch.zeros((4, 4)).int()
    # print(TorchStaticBoard.move(t_matrix, constants.UP, inplace=True))

    # print(t_matrix)
    # print(TorchStaticBoard.move(t_matrix, RIGHT, inplace=False))
    # print(t_matrix)

    # arr = Tensor([2., 0., 4., 2.])
    # print(TorchStaticBoard.collapse_array(arr, reverse=True))
    # print(arr)
    # arr = Tensor([2., 0., 4., 2.])
    # print(TorchStaticBoard.collapse_array(arr, reverse=True))
    # print(arr)

    # arr = Tensor([2., 0., 4., 2.])
    # print(TorchStaticBoard.collapse_array(arr, reverse=False))
    # print(arr)

    # arr = Tensor([2., 0., 2., 2.])
    # print(TorchStaticBoard.collapse_array(arr, reverse=True))
    # print(arr)

    # arr = Tensor([2., 0., 2., 2.])
    # print(TorchStaticBoard.collapse_array(arr, reverse=False))
    # print(arr)
