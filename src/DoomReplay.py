#!/usr/bin/python3
'''
Visual-Doom-AI: DoomReplay.py
Authors: Rafael Zamora
Last Updated: 12/15/16
CHANGE-LOG:

'''

"""
This script replays an episode of VizDoom. By default looks for replay file
stored in /data/doom_spectator_run.

Command line arguments:
-h ;help command
-f <replay_filename> ;sets replay file

"""

import sys, getopt, datetime
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution

def cmd_line_args(argv):
    '''
    cmd_line_args() is used to parse command line arguments.
    Returns list of arguments.

    '''
    filename = ''
    verbose = False
    try:
        opts, args = getopt.getopt(argv[1:], "hvf:")
    except getopt.GetoptError:
        print("Error: DoomReplay.py -f <replay_filename>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("DoomReplay.py -f <replay_filename>")
            sys.exit()
        elif opt in ('-f'):
            filename = arg
        elif opt in ('-v'):
            verbose = True
    args = [filename, verbose]
    return args

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    filename = args[0]
    verbose = args[1]
    if filename[:2] == 'ai': path = "../data/doom_ai_run/"
    else: path = "../data/doom_spectator_run/"

    game = DoomGame()
    game.load_config("configs/doom2_singleplayer.cfg")
    game.set_screen_resolution(ScreenResolution.RES_800X600)
    game.init()

    action_history = np.genfromtxt(path + filename[:-3] + "csv", delimiter=',')

    game.replay_episode(path + filename)
    while not game.is_episode_finished():
        state = game.get_state()
        if verbose and state.number < len(action_history):
            print(state.number-1, "Actions: ", action_history[state.number-1])
        game.advance_action()
    game.close()
