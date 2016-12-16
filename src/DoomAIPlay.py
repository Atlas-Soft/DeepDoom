#!/usr/bin/python3
'''
Visual-Doom-AI: DoomAIPlay.py
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
from time import sleep
from random import choice
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution

def cmd_line_args(argv):
    '''
    cmd_line_args() is used to parse command line arguments.
    Returns list of arguments.

    '''

    doom_map = "map01"
    try:
        opts, args = getopt.getopt(argv[1:], "hm:p:")
    except getopt.GetoptError:
        print("Error: DoomAIPlay.py -m <doom_map>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("DoomSpectatorPlay.py -m <doom_map>")
            sys.exit()
        elif opt in ('-m'):
            doom_map = arg
    args = [doom_map]
    return args

def get_actions(num_of_actions):
    actions = []
    for i in range(num_of_actions):
        action = []
        for j in range(num_of_actions):
            if i == j: action.append(1)
            else: action.append(0)
        actions.append(action)
    return actions

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    doom_map = args[0]
    iterations = 1000
    date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    filename = "ai" + "_" + doom_map + "_" + date + ".lmp"

    game = DoomGame()
    game.load_config("configs/doom2_singleplayer.cfg")
    game.set_doom_map(doom_map)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_episode_timeout(iterations)
    game.init()

    sleep_time = 1.0 / 60.0
    action_history = []
    actions = get_actions(game.get_available_buttons_size())

    game.new_episode("../data/doom_ai_run/" + filename)
    while not game.is_episode_finished():

        state = game.get_state()
        action_history.append(game.get_last_action())

        n = state.number
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer

        tics = 1
        game.set_action(choice(actions))
        game.advance_action(tics)

        if sleep_time > 0: sleep(sleep_time)
    action_history.append(game.get_last_action())
    game.close()

    action_history = np.array(action_history)
    np.savetxt("../data/doom_ai_run/" + filename[:-3] + "csv", action_history, fmt='%i', delimiter=",")
    print("Replay Filename: ", filename)
