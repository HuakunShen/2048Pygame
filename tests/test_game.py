from typing import List
from src.game.controller.game import Game
from src.agent.agentImpl import BacktrackingAIPlayer, RandomGuessAIPlayer

# global variables
backtrack_search_depth = 0
random_search_per_move = 20
random_search_length = 10


def get_player(g: Game, player_type: str):
    global backtrack_search_depth
    global random_search_per_move
    global random_search_length
    if player_type == "backtracking":
        return BacktrackingAIPlayer(game=g, search_depth=backtrack_search_depth, quiet=True, ui=False)
    elif player_type == "random":
        return RandomGuessAIPlayer(
            game=g, searches_per_move=random_search_per_move, search_length=random_search_length, quiet=True, ui=False)
    else:
        raise ValueError("wrong player type")


def _test_player(seeds: List[int], scores: List[int], player_type: str):
    for i in range(len(seeds)):
        g = Game(seed=seeds[i], goal=2048)
        player = get_player(g, player_type)
        score, max_val, runtime = player.run()
        assert score == scores[i] and max_val == 2048


class TestGame:
    def test_random_guess_player(self):
        seeds = [44, 81]
        scores = [20300, 20224]
        _test_player(seeds, scores, player_type="random")

    def test_backtracking_ai_player(self):
        global backtrack_search_depth
        # depth 2
        seeds = [52, 86, 91, 97]
        scores = [20180, 20336, 20612, 20172]
        backtrack_search_depth = 2
        _test_player(seeds, scores, player_type="backtracking")
        # depth 3
        seeds = [0, 98]
        scores = [20344, 20356]
        backtrack_search_depth = 3
        _test_player(seeds, scores, player_type="backtracking")
