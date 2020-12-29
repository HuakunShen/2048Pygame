import torch
import random
import constants
import numpy as np
import pandas as pd
from numba import njit
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
from constants import UP, DOWN, LEFT, RIGHT


class StaticBoard(ABC):
    seed = constants.SEED

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


class NumpyStaticBoard(StaticBoard):
    seed = constants.SEED

    @staticmethod
    def set_random_seed(seed: int = 2048):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_pd_df(matrix: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(data=matrix).astype(int)

    @staticmethod
    @njit
    def get_empty_coordinates(matrix: np.ndarray) -> np.ndarray:
        return np.argwhere(matrix == 0)

    @staticmethod
    @njit
    def has_empty_cell(matrix: np.ndarray) -> bool:
        return matrix.min() == 0

    @staticmethod
    def has_won(matrix: np.ndarray, goal: int = 2048) -> bool:
        return goal in matrix

    @staticmethod
    @njit
    def get_max_val(matrix: np.ndarray) -> int:
        return matrix.max()

    @staticmethod
    def get_random_empty_cell_coordinate(matrix: np.ndarray) -> Union[np.array, None]:
        empty_cell_coordinates = np.argwhere(matrix == 0)
        return None if len(empty_cell_coordinates) == 0 else empty_cell_coordinates[
            np.random.randint(0, len(empty_cell_coordinates))]

    @staticmethod
    def get_empty_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        return np.zeros((width, height)).astype('int64')

    @staticmethod
    def set_random_cell(matrix: np.ndarray, inplace: bool = True) -> Tuple[np.ndarray, bool]:
        empty_cell_coordinates = np.argwhere(matrix == 0)
        empty_cell = None if len(empty_cell_coordinates) == 0 else empty_cell_coordinates[
            np.random.randint(0, len(empty_cell_coordinates))]
        if not inplace:
            matrix = matrix.copy()
        if empty_cell is None:
            return matrix, False
        else:
            is_4 = random.random() < 0.1
            matrix[empty_cell[0], empty_cell[1]] = 4 if is_4 else 2
            return matrix, True

    @staticmethod
    def get_init_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        mesh_grid = np.meshgrid(np.arange(width), np.arange(height))
        coordinates = np.stack(mesh_grid, axis=2).reshape(width * height, 2)
        rand_coordinates = coordinates[np.random.choice(
            len(coordinates), 2, replace=False)]
        matrix = np.zeros((width, height))
        for i in range(2):
            matrix[rand_coordinates[i][0], rand_coordinates[i]
            [1]] = 2
        return matrix

    @staticmethod
    def compute_is_done(matrix: np.ndarray) -> bool:
        h, w = matrix.shape
        for row_i in range(h):
            row = matrix[row_i, :]
            for col_i in range(len(row)):
                if matrix[row_i, col_i] == 0:
                    return False
                neighbor_coordinates = NumpyStaticBoard.get_neighbors_coordinates(
                    matrix, row_i, col_i)
                for neighbor in neighbor_coordinates:
                    if matrix[neighbor[0], neighbor[1]] == matrix[row_i, col_i]:
                        return False
        return True

    @staticmethod
    @njit
    def collapse_array(arr: np.ndarray, reverse: bool = False) -> np.array:
        arr_len = len(arr)
        changed = False
        score = 0
        has_merged_arr = np.zeros_like(arr)
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

    @staticmethod
    def move(matrix: np.ndarray, direction: Union[UP, DOWN, LEFT, RIGHT], inplace: bool = True) -> Tuple[
        np.ndarray, int, bool]:
        score = 0
        changed = False
        if not inplace:
            matrix = matrix.copy()
        for i in range(len(matrix)):
            if direction == UP:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[:, i], reverse=True)
            elif direction == DOWN:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[:, i], reverse=False)
            elif direction == LEFT:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[i, :], reverse=True)
            elif direction == RIGHT:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[i, :], reverse=False)
            else:
                raise ValueError(f"Invalid direction: {direction}")
            changed = max(changed, changed_)
            score += score_
        return matrix, score, changed


class TorchStaticBoard(object):
    seed = constants.SEED

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

    @staticmethod
    def get_max_val(matrix: Tensor):
        return int(torch.max(matrix))

    @staticmethod
    def get_random_empty_cell_coordinate(matrix: Tensor) -> Union[Tensor, None]:
        empty_cell_coordinates = TorchStaticBoard.get_empty_coordinates(matrix)
        return None if len(empty_cell_coordinates) == 0 else empty_cell_coordinates[
            np.random.randint(0, len(empty_cell_coordinates))]

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
        score = 0
        changed = False
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


if __name__ == '__main__':
    mat = np.array([
        [0, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 2, 4, 0],
        [2, 0, 0, 2]
    ])
    print(mat)
    print(NumpyStaticBoard.set_random_cell(mat))
    print(NumpyStaticBoard.get_random_empty_cell_coordinate(mat))
