import itertools
import numpy as np
import torch
import torch.utils.data
import pandas as pd

# TODO: pinned memory for fast GPU data transfer
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_file='results.csv', game_ids=None, transform=None, num_workers=0):
        """
        Parameters
        ----------
        path: str or pathlib.PosixPath
            Path to dataset root directory. This directory contains all played games.

        Returns
        -------
        state: torch.Tensor
        label: tuple(search_probabilities, winner)
        """
        self.transform = transform # not implemented
        self.data = []

        sort_key = lambda i: int(i.name.split('_')[1].split('.')[0]) # pull timestep from filename: N+_N+.suffix

        # build list of filepaths to load. must be sorted to matchlabels with inputs by index
        csv = pd.read_csv(path/csv_file)
        if game_ids is None:
            game_ids = csv.game_id.values

        for game_id in game_ids:
            winner = csv[csv.game_id==game_id].winner.values.astype(int)[0] # array → int
            flist_search = sorted([fp for fp in (path/'search_probabilities').iterdir() if fp.name.split('_')[0]==str(game_id)], key=sort_key)
            flist_states = sorted([fp for fp in (path/'states').iterdir() if fp.name.split('_')[0]==str(game_id)], key=sort_key)
            assert len(flist_search) == len(flist_states), f"File count mismatch: Game {game_id} has missing files: search probabilities: {len(flist_search)}, states: {len(flist_states)}."
            labels = list(zip(flist_search, itertools.cycle((winner,))))
            self.data += list(zip(flist_states, labels))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fpath_state = self.data[idx][0]
        fpath_search,winner = self.data[idx][1]
        state = torch.from_numpy(np.load(fpath_state)).type(torch.float32) # TODO: handle tensor typing w/ transforms
        search_probabilities = abs(torch.from_numpy(np.load(fpath_search))).type(torch.float32) # NOTE: ensure correctly signed values are saved to dataset
        winner = torch.tensor([winner]).type(torch.float32)
        
        return state, (search_probabilities, winner)
