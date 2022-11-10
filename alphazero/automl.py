from pathlib import Path
import tqdm

from models import GoNet
from train import loss_function
from search import MCSTree
from recorder import record_game
from data_utils import Dataset
from train import fit
from selfplay import compare_models

import torch
import torch.utils.data
import numpy as np
import pandas as pd

# automl pipeline:
# scan for and select latest model
# next model = copy(model)
# for n iterations:
#     datagen
#     train next model
#     compare next model vs. model
#     if winrate ≥ 55%:
#         model = next model
#         next model = copy(model)
#         update dataset path

if __name__ == '__main__':
    n_sims = 50
    n_actions = 362
    batch_size = 48

    n_iterations = 10
    n_games = 20
    n_records = 20

    serial_number = 0

    # preferred architecture (automl will select latest serial number regardless of arch if null)
    arch_type = None

    # model file format: level_archtype_serialnumber.pth
    #     level:          level the model corresponds to
    #     archtype:       model architecture code; allows for experiments with dfrnt model types
    #     serialnumber:   allows for experiments with different training parameters; checkpoint number

    # scan for an select latest model
    print(f"Scanning for latest model..")
    models_path = Path.home()/'alphazero_data/model_checkpoints'
    models_path.mkdir(exist_ok=True)
    models_flist = sorted([fpath for fpath in models_path.iterdir() if fpath.suffix == '.pth'], key=lambda i: i.name.split('_')[0]) # sort by level
    if arch_type is not None: # filter by archtype
        models_flist = [fpath for fpath in models_flist if fpath.name.split('_')[1] == arch_type]
    models_flist = [fpath for fpath in models_flist if fpath.name.split('_')[0] == models_flist[-1].name.split('_')[0]] # grab last level
    models_flist = sorted(models_flist, key=lambda i: i.name.split('_')[2].split('.')[0]) # sort by serial number

    if len(models_flist) == 0:
        print(f"No models found in path. Will use newly-intialized model.")
        level = 0
        model_fpath = models_path/f'{level}_rn20block_{serial_number}.pth'
        print(f"Saving model: {model_fpath}")
        model = GoNet()
        torch.save(model.state_dict(), model_fpath) # save so model scan be loaded later
    else:
        model_fpath = models_flist[-1]
        level = int(model_fpath.name.split('_')[0])
        serial_number = int(model_fpath.name.split('_')[2].split('.')[0])
        print(f"Found model: {model_fpath}")

    # initialized model and next model
    print(f"Loading model: {model_fpath}")
    model = GoNet()
    model.load_state_dict(torch.load(model_fpath))
    next_model = GoNet()
    next_model.load_state_dict(torch.load(model_fpath))
    # move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    next_model.to(device)
    print(f"Complete.")

    # initialize MCS Trees
    print(f"Initializing search trees.")
    player_1 = MCSTree(model=model, n_sims=n_sims, n_actions=n_actions)
    player_2 = MCSTree(model=model, n_sims=n_sims, n_actions=n_actions)
    print(f"Complete")

    for iter_i in range(n_iterations):
        # datagen
        csv_name = 'results.csv'
        data_dir = Path.home()/'alphazero_data'
        level_path = data_dir/f'level_{str(level)}'
        csv_fpath = level_path/csv_name

        # initialize the level's data directory if it's empty
        for path_object in [level_path, level_path/'search_probabilities', level_path/'states']:
            if not path_object.exists():
                path_object.mkdir()
        
        # load master csv file, create if doesn't exist
        if not csv_fpath.exists():
            csv = pd.DataFrame(columns=['game_id','winner'])
            csv.to_csv(csv_fpath, index=False)
        csv = pd.read_csv(csv_fpath)

        # get next game's ID number
        game_ids = csv.game_id.values
        next_game_id = game_ids.max() + 1 if len(game_ids) > 0 else 0 # type: ignore 
        print(f"{len(game_ids)} recorded level {level} games found in path.")

        # recording loop
        print(f"Generating Data from {n_records} self-play games.")
        progress_bar = tqdm.tqdm(desc="Game")
        for i in range(n_records):
            game_id = next_game_id + i
            record_game(player=player_1, path=level_path, csv_name=csv_name, game_id=game_id, batch_size=batch_size)
            player_1.reset()
            progress_bar.update()
        progress_bar.close()
        print(f"Complete.")

        # train model
        print(f"Beginning training for level {level+1} model..")

        bs = 48
        lr = 3e-3

        # get game IDs to split into datasets
        csv = pd.read_csv(csv_fpath)
        game_ids = csv.game_id.values
        valid_fraction = 0.2
        valid_n = int(len(game_ids) * valid_fraction)
        train_n = len(game_ids) - valid_n

        # random shuffle then split
        rng = np.random.default_rng()
        game_ids = rng.permutation(game_ids) # type: ignore 
        train_ids = game_ids[:train_n]
        valid_ids = game_ids[train_n:]

        # data
        ds_train = Dataset(level_path, game_ids=train_ids)
        ds_valid = Dataset(level_path, game_ids=valid_ids)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)
        dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=bs, shuffle=False)

        # loss function and optimizer
        loss_function = loss_function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train
        fit(model=next_model, epochs=1, dataloader_train=dl_train, dataloader_valid=dl_valid, 
            optimizer=optimizer, criterion=loss_function, lr=lr)
            
        player_2.model = next_model
        print(f"Complete.")

        # self-play
        print(f"Assessing new model")
        player_1.reset()
        player_2.reset()

        winrate = compare_models(player_1, player_2, n_games=n_games)

        print(f"Complete. New model win rate: {winrate*100:.2f}% -- {['Pass','Fail'][winrate < 0.55]}")

        if winrate >= .55:
            level += 1
            model_fpath = models_path/f"{level}_{model_fpath.name.split('_')[1]}_{serial_number}.pth"
            torch.save(player_2.model.state_dict(), model_fpath)
            # update old model with new model's parameters
            player_1.model.load_state_dict(torch.load(model_fpath)) # type: ignore 


