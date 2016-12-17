#!/usr/bin/python3
'''
Visual-Doom-AI: ConvertReplay.py
Authors: Rafael Zamora
Last Updated: 12/16/16
CHANGE-LOG:

'''

"""
This script converts replay files to json file with screen_buffer, depth_buffer,
and action from each state of replay. Data is base64 encoded to decrease file size.


Command line arguments:
-h ;help command
-f <replay_filename> ;sets replay file

"""

import sys, getopt, datetime, json, base64
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

def b64encode_state_data(state, action):
    state_data = {}
    state_data['screen_buffer'] = [str(state.screen_buffer.dtype),base64.b64encode(state.screen_buffer).decode('utf-8'),state.screen_buffer.shape]
    state_data['depth_buffer'] = [str(state.depth_buffer.dtype),base64.b64encode(state.depth_buffer).decode('utf-8'),state.depth_buffer.shape]
    state_data['action'] = [str(action.dtype),base64.b64encode(action).decode('utf-8'),action.shape]
    return json.dumps(state_data)

def b64decode_state_data(json_dump):
    state_data = json.loads(json_dump)
    state_data['screen_buffer'] = np.frombuffer(base64.b64decode(state_data['screen_buffer'][1]),np.dtype(state_data['screen_buffer'][0])).reshape(state_data['screen_buffer'][2])
    state_data['depth_buffer'] = np.frombuffer(base64.b64decode(state_data['depth_buffer'][1]),np.dtype(state_data['depth_buffer'][0])).reshape(state_data['depth_buffer'][2])
    state_data['action'] = np.frombuffer(base64.b64decode(state_data['action'][1]),np.dtype(state_data['action'][0])).reshape(state_data['action'][2])
    return state_data

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    filename = args[0]
    verbose = args[1]
    if filename[:2] == 'ai': path = "../data/doom_ai_run/"
    else: path = "../data/doom_spectator_run/"

    game = DoomGame()
    game.load_config("configs/doom2_singleplayer.cfg")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.init()

    action_history = np.genfromtxt(path + filename[:-3] + "csv", delimiter=',')
    action_history = action_history.astype(int)

    with open("../data/doom_buffer_action/" + filename[:-3] + "json", "w") as f:
        game.replay_episode(path + filename)
        while not game.is_episode_finished():
            state = game.get_state()
            if state.number < len(action_history):
                f.write(b64encode_state_data(state,action_history[state.number-1]) + "\n")
            game.advance_action()
        game.close()
    print("State Data Filename: ", filename[:-3] + "json")
