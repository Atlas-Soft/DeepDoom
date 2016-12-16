#!/usr/bin/python3
'''
Visual-Doom-AI: DoomSpectatorPlay.py
Authors: Rafael Zamora
Last Updated: 12/15/16
CHANGE-LOG:

'''

"""
This script runs a Spectator episode of VizDoom. The Vizdoom replay file is
stored in /data/doom_spectator_run.

Command line arguments:
-h ;help command
-m <doom_map> default=map01 ;sets Doom level
-p <player_name> default=player ;sets player name used to denote replay file

"""

import sys, getopt, datetime
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution

def cmd_line_args(argv):
    '''
    cmd_line_args() is used to parse command line arguments.
    Returns list of arguments.

    '''

    doom_map = "map01"
    player_name = "player"
    try:
        opts, args = getopt.getopt(argv[1:], "hm:p:")
    except getopt.GetoptError:
        print("Error: DoomSpectatorPlay.py -m <doom_map> -p <player_name>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("DoomSpectatorPlay.py -m <doom_map> -p <player_name>")
            sys.exit()
        elif opt in ('-m'):
            doom_map = arg
        elif opt in ('-p'):
            player_name = arg
    args = [doom_map, player_name]
    return args

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    doom_map = args[0]
    player_name = args[1]
    date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    filename = player_name + "_" + doom_map + "_" + date + ".lmp"

    game = DoomGame()
    game.load_config("configs/doom2_singleplayer.cfg")
    game.set_doom_map(doom_map)
    game.set_mode(Mode.SPECTATOR)
    game.set_screen_resolution(ScreenResolution.RES_800X600)
    game.init()

    action_history = []

    game.new_episode("../data/doom_spectator_run/" + filename)
    while not game.is_episode_finished():
        action_history.append(game.get_last_action())
        game.advance_action()
    action_history.append(game.get_last_action())
    game.close()

    action_history = np.array(action_history)
    np.savetxt("../data/doom_spectator_run/" + filename[:-3] + "csv", action_history, fmt='%i', delimiter=",")
    print("Replay Filename: ", filename)
