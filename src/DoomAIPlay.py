#!/usr/bin/python3
'''
Visual-Doom-AI: DoomAIPlay.py
Authors: Rafael Zamora
Last Updated: 12/15/16
CHANGE-LOG:

'''

"""
This script runs AI episode of VizDoom. The Vizdoom replay file is
stored in /data/doom_ai_run.

Command line arguments:
-h ;help command
-m <doom_map> default=map01 ;sets Doom level
-i <iterations> default=1000 ;sets number of cycles
-d default=False ;sets display visible

"""

import sys, getopt, datetime, itertools
from time import sleep
from random import choice
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution, GameVariable
from VisualDoomAI import VisualDoomAI

def cmd_line_args(argv):
    '''
    cmd_line_args() is used to parse command line arguments.
    Returns list of arguments.

    '''

    doom_map = "map01"
    iterations = 1000
    display = False
    try:
        opts, args = getopt.getopt(argv[1:], "dhm:i:")
    except getopt.GetoptError:
        print("Error: DoomAIPlay.py -m <doom_map;sets Doom map> -i <iterations;sets number of cycles> -d <;sets display visible>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("DoomAIPlay.py -m <doom_map;sets Doom map> -i <iterations;sets number of cycles> -d <;sets display visible>")
            sys.exit()
        elif opt in ('-m'):
            doom_map = arg
        elif opt in ('-d'):
            display = True
        elif opt in ('-i'):
            iterations = int(arg)
    args = [doom_map, display, iterations]
    return args

def get_actions(num_of_actions):
    actions = list(itertools.product(range(2), repeat=num_of_actions))
    return actions

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    doom_map = args[0]
    display = args[1]
    iterations = args[2]
    date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    filename = "ai" + "_" + doom_map + "_" + date + ".lmp"

    game = DoomGame()
    game.load_config("configs/doom2_singleplayer.cfg")
    game.set_doom_map(doom_map)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(display)
    game.set_episode_timeout(iterations)
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.POSITION_Z)
    game.init()

    ai = VisualDoomAI()

    sleep_time = 1.0 / 60.0
    action_history = []
    actions = get_actions(game.get_available_buttons_size())

    game.new_episode("../data/doom_ai_run/" + filename)
    while not game.is_episode_finished():
        state = game.get_state()
        action_history.append(game.get_last_action())

        buffers = np.asarray(np.dstack((np.transpose(state.screen_buffer, (1, 2, 0)),state.depth_buffer)), order='C')
        ai_action = ai.act(buffers, actions)

        tics = 1
        game.set_action(list(ai_action))
        game.advance_action(tics)

        if sleep_time > 0: sleep(sleep_time)
    action_history.append(game.get_last_action())
    game.close()

    action_history = np.array(action_history)
    np.savetxt("../data/doom_ai_run/" + filename[:-3] + "csv", action_history, fmt='%i', delimiter=",")
    print("Replay Filename: ", filename)
