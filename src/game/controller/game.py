import copy
import random
import numpy as np
from torch import Tensor
from typing import Tuple, Union
from src.game.model.staticboardImpl import StaticBoard, NumpyStaticBoard
from src.game.utils import SEED, DEFAULT_GOAL, ARROW_KEYS, UP, DOWN, LEFT, RIGHT, K_r, K_q


class Game(object):
    """
    Game Class containing the state of game and methods for controlling the game.
    """

    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, width: int = 4, height: int = 4,
                 seed: int = SEED, goal: int = DEFAULT_GOAL,
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
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.matrix = matrix if matrix is not None else self.static_board.get_init_matrix(
            width, height)
        self.score = 0
        self.is_done = False
        self.width = width
        self.height = height

    def set_seed(self, seed: int = SEED) -> None:
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
                action = ARROW_KEYS[np.random.randint(0, 4)]
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
