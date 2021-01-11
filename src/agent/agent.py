import time
from abc import ABC, abstractmethod
from typing import Union, Tuple

from src.game.controller.game import Game
from src.game.utils import UP, DOWN, LEFT, RIGHT
from src.game.view.gameUI import GameUI


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
