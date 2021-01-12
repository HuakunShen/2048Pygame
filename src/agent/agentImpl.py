import numpy as np
from typing import Union
from src.agent.agent import Player
from src.game.controller.game import Game
from src.game.utils import ARROW_KEYS, UP, DOWN, LEFT, RIGHT
from src.game.model.staticboardImpl import NumpyStaticBoard, TorchStaticBoard


class BacktrackingAIPlayer(Player):
    """
    With search depth=5
    accuracy is around 33%
    in 100 seeds, the following are passing
            score  max_val     runtime
    seed
    6     20328   2048.0  247.989939
    8     20396   2048.0  260.319733
    10    20368   2048.0  243.177021
    14    20328   2048.0  241.503823
    15    20168   2048.0  246.624875
    17    20264   2048.0  238.900709
    23    20288   2048.0  249.540942
    28    20228   2048.0  249.040510
    30    20256   2048.0  245.020730
    32    20300   2048.0  246.836974
    33    20280   2048.0  245.686801
    37    20336   2048.0  256.914075
    40    20332   2048.0  255.554091
    45    20200   2048.0  246.123966
    47    20300   2048.0  250.852761
    56    20256   2048.0  253.074194
    62    20252   2048.0  244.514475
    63    20212   2048.0  245.337596
    65    20440   2048.0  245.588092
    67    20256   2048.0  246.610915
    70    20200   2048.0  237.210614
    71    20484   2048.0  248.832584
    73    20456   2048.0  251.539340
    78    20216   2048.0  247.156880
    79    20196   2048.0  235.604769
    80    20336   2048.0  242.382276
    81    20312   2048.0  252.015181
    83    20436   2048.0  253.451177
    84    20352   2048.0  242.617993
    87    20252   2048.0  243.486014
    88    20200   2048.0  242.957299
    94    20304   2048.0  210.975342
    95    20248   2048.0  177.931139
    """

    def __init__(self, game: Game, search_depth: int = 10, quiet: bool = False,
                 ui: bool = True, torch=False):
        super().__init__(game, quiet, ui)
        if not self._quiet:
            print("Init Backtracking AI Player")
        self.search_depth = search_depth
        self.board = TorchStaticBoard if torch else NumpyStaticBoard

    def recurse_tree(self, matrix: np.ndarray, depth: int) -> int:
        if depth == self.search_depth:
            return 0
        score = 0
        for i, move in enumerate(ARROW_KEYS):
            result_matrix, curr_score, changed = self.board.move(
                matrix=matrix, direction=move, inplace=False)
            self.board.set_random_cell(result_matrix, inplace=True)
            score = max(score, curr_score +
                        self.recurse_tree(result_matrix, depth + 1))
        return score

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                action=first_move, inplace=True)
            if changed:
                scores[first_move_i] += score
            else:
                continue
            score = self.recurse_tree(game_clone1.get_matrix(), 0)
            scores[first_move_i] += score
        return ARROW_KEYS[np.argmax(scores)]


class RandomGuessAIPlayer(Player):
    """
    Around 4% accuracy
    Sample:
    seed_ = 44
    g = Game(seed=seed_)
    player = RandomGuessAIPlayer(
        game=g, searches_per_move=20, search_length=10, ui=False)
    score_, max_val, runtime = player.run()

    Total Time Taken: 0:03:15.620000
    Average Time Taken: 1.96s
        score  max_val    runtime
    seed
    44    20300   2048.0  36.838138
    65    20532   2048.0  34.245618
    72    20324   2048.0  35.080057
    81    20224   2048.0  36.138344
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
            first_move = ARROW_KEYS[first_move_i]
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
        return ARROW_KEYS[np.argmax(scores)]
