from pathlib import Path
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import tqdm

from data_utils import Dataset
from models import GoNet

# AlphaZero pseudocode supplement (loss function):
# src: https://www.science.org/doi/10.1126/science.aar6404 → Supplementary Material → aar6404._datas1.zip → pseudocode.py
# download at: https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404_datas1.zip
# 
# tf.losses.mean_squared_error(value, target_value)
# +
# tf.nn.softmax_cross_entropy_with_logits(logits=policy, target=target_policy)
# 
# for weights in network.get_weights():
#     loss += weight_decay * tf.nn.l2_loss(weights)
# 
# NOTE: make sure softmax isn't calculated twice if the policy head output returns a softmax.

def loss_function(preds, labels):
    """
    Mean Squared Error + Cross Entropy Loss. Implements: loss = (winner - predicted_winner)**2 - search_probabilities * log(predicted_probabilities)

    From the AlphaGo Zero paper: l = (v_label - v)**2 - p_label * log(p)

    Doesn't include L2 regularization from paper.
    """
    search_prob_preds, value_preds = preds
    search_prob_label, value_label = labels

    # value loss, Mean Squared Error
    loss_value = F.mse_loss(value_preds, value_label)

    # policy loss, Cross Entropy (NOTE: applies log_softmax to input)
    loss_policy = F.cross_entropy(search_prob_preds, search_prob_label)

    return (loss_value + loss_policy).mean() # NOTE: should there be a mean here?

def fit_one_epoch(model=None, epoch_idx=0, optimizer=None, dataloader=None, criterion=None, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    running_loss = 0.0
    last_loss = 0.0

    progress_bar = tqdm.tqdm(dataloader, unit="batch")
    for i,data in enumerate(progress_bar, 0):
        progress_bar.set_description(f"epoch {epoch_idx}")
        state,labels = data

        # move batch to GPU if available
        if device.type == 'cuda':
            state = state.to(device)
            labels = (labels[0].to(device), labels[1].to(device))

        optimizer.zero_grad() # type: ignore 

        # forward pass, backward pass, optimize
        preds = model(state) # type: ignore 
        loss = criterion(preds, labels) # type: ignore 
        loss.backward()
        optimizer.step() # type: ignore 
        
        # report
        running_loss += loss.item()

        progress_bar.set_postfix(loss=running_loss/(1+i))

    last_loss = running_loss / (1+i) # type: ignore 

    return last_loss

def fit(model=None, optimizer=None, epochs=0, dataloader_train=None, dataloader_valid=None, criterion=None, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        model.train(True) # type: ignore 
        avg_loss = fit_one_epoch(model, epoch, optimizer=optimizer, dataloader=dataloader_train, criterion=criterion, lr=lr)

        # turn off gradients when reporting
        model.train(False) # type: ignore 

        with torch.no_grad():
            running_loss_valid = 0.0
            for i,valid_data in enumerate(dataloader_valid, 0): # type: ignore 
                valid_states,valid_labels = valid_data
                if device.type == 'cuda':
                    valid_states = valid_states.to(device)
                    valid_labels = (valid_labels[0].to(device), valid_labels[1].to(device))
                valid_preds = model(valid_states) # type: ignore 
                valid_loss = criterion(valid_preds, valid_labels) # type: ignore 
                running_loss_valid += valid_loss
            
            avg_loss_valid = running_loss_valid / (i+1) # type: ignore 
            print(f"Loss train/valid: {avg_loss}/{avg_loss_valid}")

# example
if __name__ == '__main__':
    path = Path.home()/'alphazero_data'

    # hyperparameters
    bs = 48
    lr = 1e-3

    # GPU / CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    # create training set on game 0, validation set on game 1
    ds_train = Dataset(path, game_ids=[0])
    ds_valid = Dataset(path, game_ids=[1])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=bs, shuffle=False)

    # model
    model = GoNet()
    model = model.to(device)

    # loss function and optimizer
    criterion = loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fit(model=model, epochs=10, dataloader_train=dl_train, dataloader_valid=dl_valid, 
        criterion=criterion, lr=lr)
