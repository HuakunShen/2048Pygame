from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from numba import njit

from src.game.utils import UP, DOWN, LEFT, RIGHT, SEED


class StaticBoard(ABC):
    seed = SEED

    @staticmethod
    @abstractmethod
    def get_pd_df(matrix: np.ndarray) -> pd.DataFrame:
        """
        Turn a matrix into a pandas DataFrame

        :param matrix: numpy 2d array representing a matrix
        :return: A pandas DataFrame representing a matrix
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_empty_coordinates(matrix: np.ndarray) -> np.ndarray:
        """
        Get empty cells coordinates from a matrix
        empty cell is represented by 0, so this method finds the 2D coordinates of all 0 cells
        :param matrix: a 2d integer numpy array representing a matrix, where 0 represents empty cell
        :return: an array of coordinates (2D array), shape=(nx2)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def has_empty_cell(matrix: np.ndarray) -> bool:
        """
        check if a matrix contains any empty cell (contains any 0)
        :param matrix: a 2d integer numpy array representing a matrix, where 0 represents empty cell
        :return: bool, whether a matrix has an empty cell (whether a matrix has a 0 in it)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def has_won(matrix: np.ndarray, goal: int = 2048) -> bool:
        """
        returns whether this matrix wins
        :param matrix: a 2d integer numpy array representing a matrix
        :param goal: goal to win, default=2048
        :return: bool, whether the matrix reaches the goal
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_max_val(matrix: np.ndarray) -> int:
        """
        get maximum value from matrix
        :param matrix: a 2d integer numpy array representing a matrix
        :return: maximum value from given matrix
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_empty_cell_coordinate(matrix: np.ndarray) -> Union[np.array, None]:
        """
        Get a coordinates from a random empty cell
        :param matrix: a 2d integer numpy array representing a matrix
        :return: a 1D coordinate numpy array
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_empty_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        """
        Get an empty matrix, all zeros
        :param width: int: width of matrix
        :param height: int: height of matrix
        :return: empty 2D array full of 0
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_init_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        """
        Get an init matrix, an empty matrix (all 0) with 2 random cells contains value 2
        :param width: int: width of matrix
        :param height: int: height of matrix
        :return: 2D array containing all 0 but 2 cells of value 2
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_random_cell(matrix: np.ndarray, inplace: bool = True) -> Tuple[np.ndarray, bool]:
        """
        Set a random cell within a matrix if there is any empty cell left
        :param matrix: a 2d integer numpy array representing a matrix
        :param inplace: set the random cell inplace, if False, make a copy then make changes, else make changes to original matrix
        :return: resulting matrix, a boolean whether a change has been made to matrix (if there is no empty cell, then no change is made)
        """
        raise NotImplementedError

    @staticmethod
    @njit
    def get_neighbors_coordinates(matrix: np.ndarray, row_i: int, col_i: int) -> np.ndarray:
        """
        Given a coordinate, get the coordinates of the neighboring cells
        - a regular cell has 4 neighboring coordinates
        - a cell on edge has 3 neighboring coordinates
        - a cell in corner has 2 neighboring coordinates
        :param matrix: a 2d integer numpy array representing a matrix
        :param row_i: row index of coordinate in matrix
        :param col_i: column index of coordinate in matrix
        :return: an array of coordinates (2D array), shape=(nx2)
        """
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
        return np.array(result)

    @staticmethod
    @abstractmethod
    def compute_is_done(matrix: np.ndarray) -> bool:
        """
        Given a 2D matrix, compute whether the game is over
        :param matrix: a 2d integer numpy array representing a matrix
        :return: bool: where game is over, True if game over, False otherwise
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def move(matrix: np.ndarray, direction: Union[UP, DOWN, LEFT, RIGHT], inplace: bool = True) -> Tuple[
        np.ndarray, int, bool]:
        """
        Given a matrix, make a move according to given direction
        :param matrix: a 2d integer numpy array representing a matrix
        :param direction: direction of movement (UP, DOWN, LEFT, RIGHT)
        :param inplace: bool: whether changes are made inplace, if False then make a copy of the matrix
        :return: Tuple[resulting matrix, score, changed(bool)]
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def collapse_array(arr: np.ndarray, reverse=False) -> np.array:
        raise NotImplementedError