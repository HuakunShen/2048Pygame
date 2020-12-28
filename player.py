import sys
import time
from abc import ABC, abstractmethod
from typing import Union

import pygame

import constants
import numpy as np
from constants import UP, DOWN, LEFT, RIGHT
from game import Game, GameUI


class Player(ABC):
    def __init__(self, game: Game):
        self._game = game

    @abstractmethod
    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        raise NotImplementedError

    @abstractmethod
    def run(self, fps: int = 30) -> None:
        raise NotImplementedError


class RandomGuessAIPlayer(Player):
    def __init__(self, game: Game, searches_per_move: int = 20, search_length: int = 10):
        super().__init__(game)
        print("init random guesser AI player")
        self.searches_per_move = searches_per_move
        self.search_length = search_length

    def get_move(self):
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = constants.ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(direction=first_move, inplace=True)
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

    def run(self, fps: int = 30) -> None:
        ui = GameUI(game=self._game, fps=fps)
        self._game.restart()
        ui.update_ui()
        iter = 0
        t_0 = time.time()
        while not self._game.get_is_done():
            move = self.get_move()
            self._game.move(move)
            ui.update_ui()
            if self._game.get_is_done():
                max_val = self._game.get_max_val()
                print(f"done, {iter} iters, score: {self._game.get_score()}, {round(time.time() - t_0, 2)}s, max_val: {max_val}")
                break
            if self._game.has_won():
                max_val = self._game.get_max_val()
                print(f"won, {iter} iters, score: {self._game.get_score()}, {round(time.time() - t_0, 2)}s, max_val: {max_val}")
                break
            iter += 1
        return


if __name__ == '__main__':
    g = Game(seed=62)
    player = RandomGuessAIPlayer(game=g, searches_per_move=40, search_length=20)
    player.run(fps=100)
    time.sleep(10)
    pygame.quit()
    sys.exit(0)
