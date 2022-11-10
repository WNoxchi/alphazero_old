import selfplay
from search import MCSTree
from models import GoNet
import selfplay
import torch

if __name__ == '__main__':
    print(f"Starting AlphaZero self-play demo for Go.")
    print("""
####################################################################################################
#########                             AlphaZero Self-Play Demo                             #########
#########                              © Wayne Polatkan. 2022                              #########
#  .   ...         .    .        .                ..       .       .    .                          #
#     .   ..  .                    .  .   ..      .      . .. .          .      .                  #
#  .  . .    ..  ...       . ..    .          . .  .                           .    .  .      .    #
#                           .              .     .    .  ...               .        .       . .. ..#
#.     .      .   . . .     .   .. . ..      .    .     .. .  .     .  ..         ..  :   .  .     #
#  .  .:+.   ..MM ...  ... ..      .  .  . . ...   . ..    . . ..          .  ..    .M=  . ..O     #
#  .    ~OO$   .MMM   .          .    .   ..  .     .    ... ..           . . ..  .MMM   .OOO  ..  #
#  .     :OOOOO.NMMMM       . .  . .  . .             .                .   .     MMMM..OOOOO. . .  #
#        .~OOOOOOOMMMM8            .              . .            .         .  .MMMMNOOOOOOO.   . . #
#   ..     =OOOOOOOONMMM7   .      .                       .          .  .   MMMNDOOOOOOOO.       .#
# .. .. ~.  =OOOOOOOODNNMM~  ..    ..              .  . ..                 MMNNNOOOOOOOOO   ,.     #
#... .   .777$OOOOOOOOONNNNN,              ..     .     . ... .      .   NNDNNNOOOOOOOOZ$77$       #
#    .  . .77777777OOOOONNNDNN   .   ..   .   . ..    .          .  .  .NDNNNNOOOOZ7777777,       .#
# .    . ...,777777777$$ONNDNND. .  ..   ....     .     .  . .   .    ?NNDNNDZ$7777777777...  .    #
# .     .     7777777777$Z$$ZDNN?.      .  . .              .  .     ?DNOZZ$Z$777777777I  .  ..    #
# .        ~++~?7777777777ZZZZ$$$?      .  .               .     . .?ZZZ$$$Z777777777I~=++.        #
#.   .    .. ~+++++++++++I7$ZZZZZ$?.     .  . .    .            . ??ZZZZ$Z77I+++++++++++..         #
#  .        .. =+++++++++++IIIIIIII??       .      . .        .  ??IIIIIIII+++++++++++.       .    #
#        .    .  ++++++++++++IIIIIIIII .   .             .    .,I?IIIIII?+++++++++++,       .      #
#     .     . .=============+????????ZI,..        .       ..  7IZ????????=============~    .    .  #
#. .     . .     .=============????+++ZZ7  ....  .     .   ..7Z$+++???==============    .. .  .    #
# ...      .  ..    .=====~~~~~=++++++=$ZZ,   .   . . . .  ZZ$Z+++++++~~~~~======. .  . . .  ......#
#   .      .   . .,~~~~~~~~~~~~~~=======N8OOO.     .    :OO8D$======~~~~~~~~~~~~~~=.              .#
#               .  .  .~~~~~~::::::~=~~~~:8NDNDZ7I:=77DDDNN+:~~~~~:::::::~~~~~~.     .        .  . #
#         ..     .     .,:::::::::::::~:::::7N8D8O7788DDN::::::~::::::::::::~.    ..             . #
#      .   .       .     .,::::::::::,:::::,,,7NNNMMNNN,,,::::,:,::::::::::.. .  .                 #
#          .         .     .:::::::,,,,,,,,,,,::MMMMMO:,,,,,,,,,,,,:::::::.       .   .   ..    .  #
# ..          .        .   .    .,,,,,,,,...,,,:,MMMM,:,,,..,,,,,,,,,...     .  .         ..   .   #
#..  . .   . .  .  ...    .   ..  . ,,,,.......,.?MM.,:.......,,,, . ... ... . .  .   .   . ...    #
#.  ...   .     .  .     .             ....... ...O............  . . ...   .  .   .     .. .. ...  #
#         . .           ..   ..    . .   ..   ,.......,.   ..         .   .   .   .  .      ...   .#
#      ..            ..   .        .       ..,..........,     ..           . .                     #
#.       . .                       .       ,,..............   .       .    .                  .. . #
#       ...        .   ..             .     ..............   .     ..          .      .         .  #
# .      .  .         .                        ........    .  ..                                   #
#     .   ..          .       ...  .                  .    .....   .  .  .    . . ..   ...  .      #
#     . . .... .      .       . .   .     .          . . .  .....  ..   .     . .  . .     .. .    #
#.    .       .   . . .    .  ...   . .   .  .   . .        .. . .     .  ..     .... . .  . . ... #
#  .           . .   .     ...      .       . .       .      .                 .     .           . #
#  .          .  .   .    ..   . . .       .  .  .                     ..  .           .           #
#   . .                          ...            .    .                .                    .       #
# ..     .       .    . .  .      .          .   .    .    .     . .   . .            .    .       #
#      .     .  .   .      .           .      .    ..        . .           .   .           .       #
#                                                                                                  #
####################################################################################################
""")

    print(f"Loading models..", end=" ")
    player_1_model = GoNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player_1_model.to(device)
    player_1 = MCSTree(model=player_1_model, n_sims=100, n_actions=362)

    player_2_model = GoNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player_2_model.to(device)
    player_2 = MCSTree(model=player_1_model, n_sims=100, n_actions=362)

    print(f"done")

    winner = selfplay.play_game(player_1=player_1, player_2=player_2, verbose=True)

