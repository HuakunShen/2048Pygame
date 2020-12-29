import time
import constants
import numpy as np
from typing import Union, Tuple
from game import Game, GameUI
from abc import ABC, abstractmethod
from constants import UP, DOWN, LEFT, RIGHT


class Player(ABC):
    def __init__(self, game: Game, quiet: bool = False) -> None:
        self._game = game
        self._quiet = quiet

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        raise NotImplementedError

    def run(self, fps: int = 30) -> Tuple[int, int, float]:
        ui = GameUI(game=self._game, fps=fps)
        self._game.restart()
        ui.update_ui()
        iteration = 0
        t_0 = time.time()
        while not self._game.get_is_done():
            move = self.get_move()
            self._game.move(move)
            ui.update_ui()
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
    def __init__(self, game: Game, searches_per_move: int = 20, search_length: int = 10, quiet: bool = False):
        super().__init__(game, quiet)
        if not self._quiet:
            print("init random guesser AI player")
        self.search_length = search_length
        self.searches_per_move = searches_per_move

    def get_move(self):
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = constants.ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                direction=first_move, inplace=True)
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


if __name__ == '__main__':
    seed_ = 0
    g = Game(seed=seed_)
    player = RandomGuessAIPlayer(
        game=g, searches_per_move=20, search_length=10)
    score_, max_val, runtime = player.run()
