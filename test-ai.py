import time
import datetime
import pandas as pd
from src.game.controller.game import Game
from src.agent.agentImpl import BacktrackingAIPlayer, RandomGuessAIPlayer

goal = 2048
seeds = list(range(100))
# Player = RandomGuessAIPlayer
player_type = "random"


# player_type = "backtracking"


def get_player(g: Game):
    global player_type
    if player_type == "backtracking":
        return BacktrackingAIPlayer(game=g, search_depth=5, quiet=True, ui=False)
    elif player_type == "random":
        return RandomGuessAIPlayer(
            game=g, searches_per_move=20, search_length=10, quiet=True, ui=False)
    else:
        raise ValueError("wrong palyer type")


def run(seed, i=0):
    global goal
    global player_type
    g = Game(seed=seed, goal=goal)
    player = get_player(g)
    score_, max_val, runtime = player.run()
    return seed, score_, max_val, runtime


if __name__ == '__main__':
    t0 = time.time()
    df = pd.DataFrame(columns=['seed', 'score', 'max_val', 'runtime'])
    df.set_index('seed', inplace=True)

    # with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #     results = p.map(run, seeds)

    # with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #     results = list(tqdm.tqdm(p.imap(run, seeds), total=len(seeds)))

    results = [run(i) for i in seeds]

    for result in results:
        df.loc[result[0]] = {"score": result[1],
                             "max_val": result[2], "runtime": result[3]}
    print(df)
    total_time = round(time.time() - t0, 2)
    print(f"Total Time Taken: {str(datetime.timedelta(seconds=total_time))}")
    avg_time = round(total_time / len(seeds), 2)
    print(f"Average Time Taken: {avg_time}s")
    print(df[df['max_val'] == goal])
