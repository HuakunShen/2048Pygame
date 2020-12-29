import sys
import copy
import random
import constants
import pygame
from constants import UP, DOWN, LEFT, RIGHT, K_r, K_q
import numpy as np
from typing import Union
from torch import Tensor

from staticboard import TorchStaticBoard, NumpyStaticBoard, StaticBoard


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


class Game(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, width: int = 4, height: int = 4,
                 seed: int = constants.SEED, goal: int = constants.DEFAULT_GOAL,
                 static_board: StaticBoard = NumpyStaticBoard):
        self.goal = goal
        self.seed = seed
        self.static_board = static_board
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.matrix = matrix if matrix is not None else self.static_board.get_init_matrix(
            width, height)
        self.score = 0
        self.is_done = False
        self.width = width
        self.height = height

    def set_seed(self, seed: int = constants.SEED):
        self.seed = seed

    def get_matrix(self):
        return self.matrix

    def get_score(self):
        return self.score

    def get_is_done(self):
        return self.is_done

    def has_won(self):
        return self.static_board.has_won(self.matrix, self.goal)

    def get_max_val(self):
        return self.static_board.get_max_val(self.matrix)

    def restart(self):
        self.__init__(matrix=None, width=self.width,
                      height=self.height, seed=self.seed)

    def move(self, direction: Union[UP, DOWN, LEFT, RIGHT, K_r, K_q, None] = None, inplace: bool = True):
        if direction == K_r:
            self.restart()
            return self.matrix, 0, True
        else:
            if direction is None:
                direction = constants.ARROW_KEYS[np.random.randint(0, 4)]
            matrix, score, changed = self.static_board.move(
                matrix=self.matrix, direction=direction, inplace=inplace)
            self.score += score
            if changed:
                matrix, added = self.static_board.set_random_cell(
                    self.matrix, inplace=inplace)
                self.is_done = self.static_board.compute_is_done(matrix)
            return matrix, score, changed

    def clone(self):
        return copy.deepcopy(self)


class GameUI(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, game: Game = None, width: int = 800, height: int = 950,
                 margin: int = 10, fps: int = 30):
        self.fps = fps
        self.game = Game(matrix=matrix) if game is None else game
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.margin = margin
        self.block_size = (self.width - (self.game.get_matrix().shape[0] + 1) * margin) // \
            self.game.get_matrix().shape[0]
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
                rect = pygame.Rect(col_i * self.block_size + (col_i + 1) * self.margin,
                                   row_i * self.block_size + (row_i + 1) *
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
        self.clock.tick(self.fps)
        self.screen.fill(constants.BG_COLOR)
        self.draw_grid()
        self.update_score()
        self.update_msg()
        pygame.display.flip()


if __name__ == "__main__":
    # matrix = np.array([
    #     [16, 32, 16, 16],
    #     [32, 16, 32, 32],
    #     [16, 32, 16, 16],
    #     [32, 16, 32, 0]
    # ])
    game = GameUI()
    game.main()
    # np.random.seed(10)
    # g = Game(seed=10)
    # g.set_seed(10)
    # print(NumpyStaticBoard.get_pd_df(g.get_matrix()))
    # random.seed(0)
    # np.random.seed(100)
    # print(NumpyStaticBoard.get_init_matrix())
