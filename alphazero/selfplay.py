import tqdm
import torch

import state_engine
from search import MCSTree
from models import GoNet

# Self-Play module spec:
# in:   model files to compare, N games to play
# func: compare 2 models for N games
# out:  winrate of model 2

def compare_models(player_1:MCSTree, player_2:MCSTree, n_games=100, verbose=False):
    """
    Plays 2 AlphaZero AIs against each other for N games. Returns `player_2`'s winrate.
    """
    wins = 0
    progress_bar = tqdm.trange(n_games, desc="Game")
    for i in progress_bar:
        progress_bar.set_description(f"game {i+1}")

        winner = play_game(player_1, player_2, verbose=verbose)
        if winner == 1:
            wins += 1
        # reset MCSTrees after each game
        player_1.reset()
        player_2.reset()

        progress_bar.set_postfix(winrate=wins/(i+1))
    progress_bar.close()
    return wins/n_games

def play_game(player_1:MCSTree, player_2:MCSTree, verbose=True):
    """
    Plays one game.
    """
    game_end = player_1.root.state.is_game_over()

    current_player = player_1.root.state.to_play # 1:p1, -1:p2

    player_names = {1:'Black',-1:'White',0:'Draw'}

    while not game_end:
        if current_player == 1:
            move = player_1.play_move()
            player_2.play_move(move_index=move)
        else:
            move = player_2.play_move()
            player_1.play_move(move_index=move)
        
        if verbose:
            print(f"Turn: {player_1.root.state.n}")
            print(f"{player_names[current_player]} move: {state_engine.action_array_index_to_coord(move)}")  # type: ignore
            print(f"Captures B,W: {player_1.root.state.caps}")
            print(player_1.root.state)

        game_end = player_1.root.state.is_game_over()

        assert player_1.root.state.is_game_over() == player_2.root.state.is_game_over(), f"state error: player 1 and player 2 'game over' out of sync: {player_1.root.state.is_game_over()} / {player_2.root.state.is_game_over()} "
        assert player_1.root.state.to_play == player_2.root.state.to_play, f"state error: player 1 and player 2 'player turn' out of sync: {player_1.root.state.to_play} / {player_1.root.state.to_play}"
        assert player_1.root.state.result() == player_2.root.state.result(), f"state error: player 1 and player 2 'winner' out of sync: {player_1.root.state.result()} / {player_1.root.state.result()}"

        current_player = player_1.root.state.to_play

    winner = player_1.root.state.result()
    if verbose:
        print(f"Winner: {player_names[winner]}")

    return winner

# example
if __name__ == '__main__':
    player_1_model = GoNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player_1_model.to(device)
    player_1 = MCSTree(model=player_1_model, n_sims=100, n_actions=362)

    player_2_model = GoNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player_2_model.to(device)
    player_2 = MCSTree(model=player_1_model, n_sims=100, n_actions=362)

    winrate = compare_models(player_1, player_2, n_games=50)
    print(winrate)
