import re
import time
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import multiprocessing
from typing import Dict, List
from tabulate import tabulate
from src.game.controller.game import Game
from src.agent.agentImpl import BacktrackingAIPlayer, RandomGuessAIPlayer

BACKTRACKING = "backtracking"
RANDOM = "random"


def parse_seeds_arg(seeds_str: str) -> List[int]:
    """parse command line seeds arguments into a list of integer seeds

    :param seeds_str: [description]
    :type seeds_str: str
    :raises ValueError: [description]
    :return: [description]
    :rtype: [type]
    """
    if re.match(r"\d+-\d+", seeds_str):
        low, high = seeds_str.split("-")
        return list(range(int(low), int(high)))
    elif re.match(r"\d+", seeds_str):
        return [int(seeds_str)]
    else:
        raise ValueError("invalid seeds argument")


def get_player(g: Game, **kwargs):
    if kwargs["player_type"] == BACKTRACKING:
        return BacktrackingAIPlayer(game=g, search_depth=kwargs['search_depth'], quiet=True, ui=kwargs['ui'])
    elif kwargs["player_type"] == RANDOM:
        return RandomGuessAIPlayer(
            game=g, searches_per_move=kwargs['searches_per_move'],
            search_length=kwargs['search_length'],
            quiet=True, ui=kwargs["ui"])
    else:
        raise ValueError("wrong palyer type")


def run(params: Dict):
    g = Game(seed=params['seed'], goal=params['goal'])
    player = get_player(g, **params)
    score_, max_val, runtime = player.run()
    return params['seed'], score_, max_val, runtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser("2048 AI Argument parser")
    subparsers = parser.add_subparsers(help='player type', dest='player_type')
    parser_backtracking = subparsers.add_parser(BACKTRACKING, help=f'{BACKTRACKING} mode help')
    parser_backtracking.add_argument('--search_depth', type=int, default=2)
    parser_random = subparsers.add_parser(RANDOM, help=f'{RANDOM} mode help')
    parser_random.add_argument('--search_length', type=int, default=10)
    parser_random.add_argument('--searches_per_move', type=int, default=20)
    for subparser in [parser_random, parser_backtracking]:
        subparser.add_argument("--goal", default=2048, type=int, help="Goal to end game, default=2048")
        subparser.add_argument("--ui", action="store_true", help="Goal to end game")
        subparser.add_argument("--seeds", default="1-100",
                               help="seeds to run, use either a single seed or a range of seeds e.g. 1-100")
        subparser.add_argument("--mp", action="store_true", help="Use multiprocessing")
    args = parser.parse_args()
    seeds = parse_seeds_arg(args.seeds)
    t0 = time.time()
    df = pd.DataFrame(columns=['seed', 'score', 'max_val', 'runtime'])
    df.set_index('seed', inplace=True)

    params = [{**args.__dict__.copy(), **{'seed': seed}} for seed in seeds]
    if args.mp:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            results = list(tqdm(p.imap(run, params), total=len(seeds)))
    else:
        results = [run(param) for param in tqdm(params)]
    for result in tqdm(results):
        df.loc[result[0]] = {"score": result[1], "max_val": result[2], "runtime": result[3]}
    print(tabulate(df, headers='keys', tablefmt='pretty'))
    total_time = round(time.time() - t0, 2)
    avg_time = round(total_time / len(seeds), 2)
    success_df = df[df['max_val'] == args.goal]
    print(tabulate(success_df, headers='keys', tablefmt='pretty'))
    print(f"Total Time Taken: {str(datetime.timedelta(seconds=total_time))}")
    print(f"Average Time Taken: {avg_time}s")
    print(f"Accuracy: {round(len(success_df) / len(df) * 100)}%")
