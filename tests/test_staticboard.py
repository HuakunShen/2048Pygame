import pytest
from src.game import utils
import numpy as np
import pandas as pd
from src.game.model.staticboardImpl import NumpyStaticBoard


class TestNumpyStaticBoard:
    @pytest.fixture
    def matrix(self):
        return np.array([
            [0, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 2, 4, 0],
            [2, 0, 0, 2]
        ])

    def test_get_pd_df(self, matrix: np.ndarray):
        assert (pd.DataFrame(data=matrix) == NumpyStaticBoard.get_pd_df(matrix)).all().all()

    def test_get_empty_coordinates(self, matrix: np.ndarray):
        assert (np.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1],
                          [1, 2],
                          [1, 3],
                          [2, 0],
                          [2, 3],
                          [3, 1],
                          [3, 2]]) == NumpyStaticBoard.get_empty_coordinates(matrix)).all()

    def test_has_empty_cell(self, matrix: np.ndarray):
        assert NumpyStaticBoard.has_empty_cell(matrix)
        assert not NumpyStaticBoard.has_empty_cell(matrix + 1)

    def test_has_won(self, matrix: np.ndarray):
        assert not NumpyStaticBoard.has_won(matrix)
        assert NumpyStaticBoard.has_won(matrix, goal=4)
        assert NumpyStaticBoard.has_won(matrix + 2048)

    def test_get_max_val(self, matrix: np.ndarray):
        assert NumpyStaticBoard.get_max_val(matrix) == 4
        assert NumpyStaticBoard.get_max_val(np.where(matrix > 0, 2048, matrix)) == 2048

    def test_get_random_empty_cell_coordinate(self, matrix: np.ndarray):
        for i in range(100):
            np.random.seed(2048)
            assert (NumpyStaticBoard.get_random_empty_cell_coordinate(matrix) == np.array([1, 3])).all()

    def test_get_empty_matrix(self):
        matrix = NumpyStaticBoard.get_empty_matrix(width=4, height=4)
        assert matrix.dtype == 'int64'
        assert (matrix.shape == np.array([4, 4])).all()
        assert np.sum(matrix) == 0
        matrix = NumpyStaticBoard.get_empty_matrix(width=100, height=100)
        assert matrix.dtype == 'int64'
        assert (matrix.shape == np.array([100, 100])).all()
        assert np.sum(matrix) == 0

    def test_get_init_matrix(self):
        matrix = NumpyStaticBoard.get_init_matrix(width=4, height=4)
        assert np.sum(matrix > 0) == 2
        assert np.max(matrix) == 2 or np.max(matrix) == 4

    def test_set_random_cell(self):
        matrix = NumpyStaticBoard.get_empty_matrix(width=4, height=4)
        non_empty_count = np.sum(matrix > 0)
        assert non_empty_count == 0
        for i in range(4 * 4):
            NumpyStaticBoard.set_random_cell(matrix=matrix, inplace=True)
            assert np.sum(matrix > 0) == non_empty_count + 1
            non_empty_count += 1
        assert np.sum(matrix == 0) == 0

    def test_get_neighbors_coordinates(self, matrix):
        assert (NumpyStaticBoard.get_neighbors_coordinates(matrix, 0, 0) == np.array([[1, 0], [0, 1]])).all()
        assert (NumpyStaticBoard.get_neighbors_coordinates(matrix, 1, 1) == np.array([[0, 1],
                                                                                      [2, 1],
                                                                                      [1, 0],
                                                                                      [1, 2]])).all()

    def test_compute_is_done(self, matrix):
        assert not NumpyStaticBoard.compute_is_done(matrix)
        assert NumpyStaticBoard.compute_is_done(np.arange(1, 17).reshape(4, 4))

    def test_collapse_array(self):
        assert (NumpyStaticBoard.collapse_array(np.array([0, 2, 2, 0]), reverse=False)[0] == np.array(
            [0, 0, 0, 4])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([0, 2, 2, 0]), reverse=True)[0] == np.array(
            [4, 0, 0, 0])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([0, 4, 2, 0]), reverse=False)[0] == np.array(
            [0, 0, 4, 2])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([0, 4, 2, 0]), reverse=True)[0] == np.array(
            [4, 2, 0, 0])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([8, 2, 2, 0]), reverse=False)[0] == np.array(
            [0, 0, 8, 4])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([8, 2, 2, 0]), reverse=True)[0] == np.array(
            [8, 4, 0, 0])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([4, 4, 2, 2]), reverse=False)[0] == np.array(
            [0, 0, 8, 4])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([4, 4, 2, 2]), reverse=True)[0] == np.array(
            [8, 4, 0, 0])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([2, 4, 8, 16]), reverse=False)[0] == np.array(
            [2, 4, 8, 16])).all()
        assert (NumpyStaticBoard.collapse_array(np.array([2, 4, 8, 16]), reverse=True)[0] == np.array(
            [2, 4, 8, 16])).all()

    def test_move(self, matrix):
        # test inplace
        matrix_copy = matrix.copy()
        matrix_, score, changed = NumpyStaticBoard.move(matrix, direction=utils.RIGHT, inplace=True)
        assert id(matrix) == id(matrix_)
        assert (matrix_ == np.array([[0, 0, 0, 4],
                                     [0, 0, 0, 0],
                                     [0, 0, 2, 4],
                                     [0, 0, 0, 4]])).all()
        matrix = matrix_copy.copy()
        # test non-inplace
        matrix_, score, changed = NumpyStaticBoard.move(matrix, direction=utils.RIGHT, inplace=False)
        assert id(matrix) != id(matrix_)
        assert (matrix_ == np.array([[0, 0, 0, 4],
                                     [0, 0, 0, 0],
                                     [0, 0, 2, 4],
                                     [0, 0, 0, 4]])).all()
        # test 3 other directions
        matrix_, score, changed = NumpyStaticBoard.move(matrix, direction=utils.UP, inplace=False)
        assert id(matrix) != id(matrix_)
        assert (matrix_ == np.array([[2, 2, 2, 4],
                                     [0, 0, 4, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])).all()
        matrix_, score, changed = NumpyStaticBoard.move(matrix, direction=utils.DOWN, inplace=False)
        assert id(matrix) != id(matrix_)
        assert (matrix_ == np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 2, 0],
                                     [2, 2, 4, 4]])).all()
        matrix_, score, changed = NumpyStaticBoard.move(matrix, direction=utils.LEFT, inplace=False)
        assert id(matrix) != id(matrix_)
        assert (matrix_ == np.array([[4, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [2, 4, 0, 0],
                                     [4, 0, 0, 0]])).all()
