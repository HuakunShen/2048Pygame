import time
import constants
import numpy as np
from typing import Union, Tuple
from game import Game, GameUI
from abc import ABC, abstractmethod
from constants import UP, DOWN, LEFT, RIGHT
from staticboard import NumpyStaticBoard


class Player(ABC):
    def __init__(self, game: Game, quiet: bool = False, ui: bool = True) -> None:
        """
        Init function for Player Class
        :param game: Game: game object
        :param quiet: bool: quiet mode, whether you want messages to be printed
        """
        self.ui = ui
        self._game = game
        self._quiet = quiet

    @abstractmethod
    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        """
        :return: direction of movement: Union[UP, DOWN, LEFT, RIGHT]
        """
        raise NotImplementedError

    def run(self, fps: int = 30) -> Tuple[int, int, float]:
        """
        main logic for running the game
        :param fps: int: frame per second of animation
        :return: Tuple[score, max value reached, runtime used]
        """
        game_ui = GameUI(game=self._game, fps=fps) if self.ui else None
        self._game.restart()
        if self.ui:
            game_ui.update_ui()
        iteration = 0
        t_0 = time.time()
        while not self._game.get_is_done():
            move = self.get_move()
            self._game.move(move)
            if self.ui:
                game_ui.update_ui()
            if self._game.get_is_done():
                if not self._quiet:
                    print(
                        f"done, {iteration} iterations, score: {self._game.get_score()}, "
                        f"{round(time.time() - t_0, 2)}s, max value: {self._game.get_max_val()}")
                break
            if self._game.has_won():
                if not self._quiet:
                    print(
                        f"won, {iteration} iterations, score: {self._game.get_score()}, "
                        f"{round(time.time() - t_0, 2)}s, max value: {self._game.get_max_val()}")
                break
            iteration += 1
        return self._game.get_score(), self._game.get_max_val(), time.time() - t_0


class RandomGuessAIPlayer(Player):
    """
    Sample:
    seed_ = 44
    g = Game(seed=seed_)
    player = RandomGuessAIPlayer(
        game=g, searches_per_move=20, search_length=10, ui=False)
    score_, max_val, runtime = player.run()
    """

    def __init__(self, game: Game, searches_per_move: int = 20, search_length: int = 10, quiet: bool = False,
                 ui: bool = True):
        super().__init__(game, quiet, ui)
        if not self._quiet:
            print("Init Random guesser AI Player")
        self.search_length = search_length
        self.searches_per_move = searches_per_move

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = constants.ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                action=first_move, inplace=True)
            if changed:
                scores[first_move_i] += score
            else:
                continue
            max_cumulative_score = 0
            for later_moves in range(self.searches_per_move):
                move_number = 1
                game_clone2 = game_clone1.clone()
                changed = True
                score_cumulative = 0
                while changed and move_number < self.search_length:
                    matrix, score, changed = game_clone2.move(
                        inplace=True)  # make a random move
                    if changed:
                        score_cumulative += score
                        # scores[first_move_i] += score
                        move_number += 1
                max_cumulative_score = max(
                    max_cumulative_score, score_cumulative)
            scores[first_move_i] += max_cumulative_score
        return constants.ARROW_KEYS[np.argmax(scores)]


class BacktrackingAIPlayer(Player):
    def __init__(self, game: Game, search_length: int = 10, quiet: bool = False,
                 ui: bool = True):
        super().__init__(game, quiet, ui)
        if not self._quiet:
            print("Init Backtracking AI Player")
        self.search_depth = search_length

    def recurse_tree(self, matrix: np.ndarray, depth: int) -> int:
        if depth == self.search_depth:
            return 0
        score = 0
        scores1, scores2 = [], []
        for i, move in enumerate(constants.ARROW_KEYS):
            result_matrix, curr_score, changed = NumpyStaticBoard.move(
                matrix=matrix, direction=move, inplace=False)
            NumpyStaticBoard.set_random_cell(result_matrix, inplace=True)
            # scores1.append(curr_score)
            # scores2.append(self.recurse_tree(result_matrix, depth + 1))
            score += curr_score
            score += self.recurse_tree(result_matrix, depth + 1)
        # return sum(scores1) + sum(scores2)
        return score

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = constants.ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                action=first_move, inplace=True)
            if changed:
                scores[first_move_i] += score
            else:
                continue
            score = self.recurse_tree(game_clone1.get_matrix(), 0)
            scores[first_move_i] += score
        return constants.ARROW_KEYS[np.argmax(scores)]


if __name__ == '__main__':
    seed_ = 0
    g = Game(seed=seed_)
    player = BacktrackingAIPlayer(
        game=g, search_length=7, quiet=False, ui=True)
    player.run(fps=100)
