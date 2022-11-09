from search import MCSTree
from models import GoNet
import state_engine
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import tqdm
import time

def record_game(player:MCSTree, path, csv_name, game_id:int, batch_size=1):
    """
    Records one game.
    """
    csv_fpath = path/csv_name
    csv = pd.read_csv(csv_fpath)
    player.batch_size = batch_size

    # play game
    progress_bar = tqdm.tqdm(desc=f"Game no. {game_id} turn")
    while not player.root.state.is_game_over():
        # MCSTree's policy is created by latest run of MCTS.
        # policy is updated after MCTS, but its corresponding state is pruned when the new node is 
        #   chosen.
        # → timestep and state are saved before a move, policy saved after.

        # simulate - select - play
        timestep = player.root.state.n
        state = state_engine.get_history(player.root.state)
        player.play_move()
        policy = player.search_probabilities

        # record turn
        np.save(path/'state'/f'{game_id}_{timestep}', state)
        np.save(path/'search_probabilities'/f'{game_id}_{timestep}', policy)

        progress_bar.update()

    progress_bar.close()

    # record winner
    winner = player.root.state.result()
    csv_data = [[game_id, winner]]
    csv_newline = pd.DataFrame(csv_data, columns=['game_id','winner'])
    csv = pd.concat((csv, csv_newline), ignore_index=True)
    csv.to_csv(csv_fpath, index=False)

def delete_game(path, game_id):
    """
    Deletes all data pertaining to the provided `game_id`.

    Affected paths: `results.csv`, `search_probabilities/`, `states/`.
    """
    n = 0
    # delete data
    for f in Path.iterdir(path/'search_probabilities'):
        if not f.is_dir() and f.suffix == '.npy' and f.name.split('_')[0] == str(game_id):
            f.unlink()
            n += 1
    print(f"Deleted {n} probabilities.")

    for f in Path.iterdir(path/'states'):
        if not f.is_dir() and f.suffix == '.npy' and f.name.split('_')[0] == str(game_id):
            f.unlink()
            n += 1
    print(f"Deleted {n} states.")

    # delete record from master csv file
    csv = pd.read_csv(path/'results.csv')
    csv = csv.drop(csv[csv['game_id']==game_id].index)
    csv.to_csv(path/'results.csv', index=False)

def get_latest_game_id(path, csv_name='results.csv'):
    """
    Returns ID number of the last game played in a directory.

    Used to remove data from incompletely-recorded games.

    Note: if a game record didn't update the level's csv file, the game ID to delete will be 
          `game_ids.max() + 1`.
    """
    csv = pd.read_csv(path/csv_name)
    game_ids = csv.game_id.values

    if len(game_ids) == 0:
        return None
    return game_ids.max()  # type: ignore


# example
if __name__ == '__main__':
    n_games = 1
    csv_name = 'results.csv'

    # dataset root path
    data_dir = Path.home()/'alphazero_data'
    if not data_dir.exists():
        data_dir.mkdir()

    # find all level directories, and select
    level_paths = [p for p in data_dir.iterdir() if p.name.split('_')[0] == 'level' and p.is_dir()]

    # TODO parameterize level selection and creation; match to model
    if len(level_paths) == 0:
        print(f"No levels found. Creating \"{data_dir/'level_0'}\"")
        path = data_dir/'level_0'
        path.mkdir()
        (path/'search_probabilities').mkdir()
        (path/'states').mkdir()
    else:
        path = level_paths[0] # default select first level
    # specify level & directory
    level = path.name.split('_')[-1]

    # load CSV; create if doesn't exist
    csv_fpath = path/csv_name
    if not csv_fpath.exists():
        print(f"CSV file \"{csv_fpath}\" not found. Creating.")
        csv = pd.DataFrame(columns=['game_id','winner'])
        csv.to_csv(csv_fpath, index=False)
    csv = pd.read_csv(csv_fpath)

    # get game IDs
    game_ids = csv.game_id.values
    next_game_id = game_ids.max() + 1 if len(game_ids) > 0 else 0 # type: ignore

    # recording loop
    for i in range(n_games):
        game_id = next_game_id + i
        print(f"Simulating Level {level}, Game {game_id}.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GoNet()
        player = MCSTree(model=model, n_sims=100)
        player.model.to(device); # type: ignore

        t0 = time.time()
        record_game(player=player, path=path, csv_name=csv_name, game_id=game_id, batch_size=48)
        t = time.time() - t0

        print(f"Game {game_id} complete in {np.round(t)} s. {player.root.state.n + 1} timesteps.")