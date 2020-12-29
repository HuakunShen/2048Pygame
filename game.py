import sys
import copy
import random
import pygame
import constants
import numpy as np
from torch import Tensor
from typing import Union, Tuple
from constants import UP, DOWN, LEFT, RIGHT, K_r, K_q
from staticboard import TorchStaticBoard, NumpyStaticBoard, StaticBoard


def get_bg_color(val: Union[int, str]):
    """
    Get the background color of 2048 game
    :param val: value of cell, different values have different background color
    :return: a Tuple of RGB, chosen from constants.CELL_BG_COLOR_MAP
    """
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
    """
    Game Class containing the state of game and methods for controlling the game.
    """

    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, width: int = 4, height: int = 4,
                 seed: int = constants.SEED, goal: int = constants.DEFAULT_GOAL,
                 static_board: StaticBoard = NumpyStaticBoard):
        """
        Init function of Game Class
        :param matrix: predefined 2D matrix for the game, if None then a random board will be generated
        :param width: int: game board width
        :param height: int: game board height
        :param seed: int: random seed, a seed is for making the game more reproducible
        :param goal: int: goal maximum value, default is 2048
        :param static_board: StaticBoard class for manipulating matrix, can be the different classes depending on the
        matrix data type, default is NumpyStaticBoard
        """
        self.goal = goal
        self.seed = seed
        self.static_board = static_board
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.matrix = matrix if matrix is not None else self.static_board.get_init_matrix(
            width, height)
        self.score = 0
        self.is_done = False
        self.width = width
        self.height = height

    def set_seed(self, seed: int = constants.SEED) -> None:
        """
        set random seed for the game
        :param seed: int: random seed
        :return: None
        """
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def get_matrix(self) -> Union[Tensor, np.ndarray]:
        """
        Getter of the matrix
        :return: board (matrix) of the game object
        """
        return self.matrix

    def get_score(self) -> int:
        """
        Getter of score of the game
        :return: int: score
        """
        return self.score

    def get_is_done(self):
        """
        Getter of is_done of the game
        :return: bool: status of game, whether game is done
        """
        return self.is_done

    def has_won(self) -> bool:
        """
        :return: bool: whether game has won
        """
        return self.static_board.has_won(self.matrix, self.goal)

    def get_max_val(self) -> int:
        """
        :return: maximum value within game (matrix)
        """
        return self.static_board.get_max_val(self.matrix)

    def restart(self) -> None:
        """
        restart the game by calling init function, reset seed
        :return: None
        """
        self.__init__(matrix=None, width=self.width,
                      height=self.height, seed=self.seed)

    def move(self, action: Union[UP, DOWN, LEFT, RIGHT, K_r, K_q, None] = None,
             inplace: bool = True) -> Tuple[np.ndarray, int, bool]:
        """
        Make a move
        :param action: 4 directions or restart or quit, Union[UP, DOWN, LEFT, RIGHT, K_r, K_q, None]
        :param inplace: whether changes are made inplace to the matrix
        :return: matrix: 2D array, score: int, changed: bool
        """
        if action == K_r:
            self.restart()
            return self.matrix, 0, True
        else:
            if action is None:
                action = constants.ARROW_KEYS[np.random.randint(0, 4)]
            matrix, score, changed = self.static_board.move(
                matrix=self.matrix, direction=action, inplace=inplace)
            self.score += score
            if changed:
                matrix, added = self.static_board.set_random_cell(
                    self.matrix, inplace=inplace)
                self.is_done = self.static_board.compute_is_done(matrix)
            return matrix, score, changed

    def clone(self):
        """
        make a clone (deepcopy) of game
        :return: a deepcopy of game
        """
        return copy.deepcopy(self)


class GameUI(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, game: Game = None, width: int = 800, height: int = 950,
                 margin: int = 10, fps: int = 30) -> None:
        """
        Init function for GameUI
        :param matrix: predefined 2D matrix for the game, if None then a random board will be generated
        :param game: Game: game object
        :param width: int: game board width
        :param height: int: game board height
        :param margin: margin of game screen
        :param fps: int: target frame per second of animation
        :return: None
        """
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

    def run(self) -> None:
        """
        main logic for running the game for playing the game manually
        :return: None
        """
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
                                action=constants.KEY_MAP[event.key], inplace=True)
            self.update_ui()

    def set_game(self, game: Game) -> None:
        """
        Setter for game object
        :param game: Game: game object containing all game state and logic
        :return: None
        """
        self.game = game

    def _update_score(self) -> None:
        """
        update score on UI
        :return: None
        """
        font = pygame.font.Font(None, 64)
        text = font.render(
            'Score: ' + str(self.game.get_score()), 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def _update_msg(self) -> None:
        """
        update messages displayed on UI
        :return: None
        """
        font = pygame.font.Font(None, 32)
        text = font.render('Game ends, press r to restart' if self.game.get_is_done(
        ) else "Click 'q' to quit the game", True, (255, 255, 255))
        self.screen.blit(text, (50, 870))

    def _draw_grid(self) -> None:
        """
        draw the game board grid on UI
        :return: None
        """
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

    def update_ui(self) -> None:
        """
        Update UI altogether including updating score, messages displayed and grid
        :return: None
        """
        self.clock.tick(self.fps)
        self.screen.fill(constants.BG_COLOR)
        self._draw_grid()
        self._update_score()
        self._update_msg()
        pygame.display.flip()


if __name__ == "__main__":
    game = GameUI()
    game.run()
