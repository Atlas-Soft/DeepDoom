#!/usr/bin/python3
'''
Visual-Doom-AI: VisualDoomAI.py
Authors: Rafael Zamora
Last Updated: 12/16/16
CHANGE-LOG:

'''

"""
This class is used to define the Visual Doom AI.

Command line arguments:
-h ;help command
-t <train_data_filename> ;sets train data

"""
import sys, getopt
from random import choice
from ConvertReplay import b64decode_state_data
import numpy as np

class VisualDoomAI():

    def __init__(self):
        pass

    def act(self, buffers, actions):
        action = choice(actions)
        return action

    def train(self, train_data_filename):
        train_data_x = []
        train_data_y = []
        train_data_z = []
        with open("../data/doom_buffer_action/" + train_data_filename, "r") as f:
            for json_dump in f:
                state_data = b64decode_state_data(json_dump)
                train_data_x.append(state_data['buffers'])
                train_data_y.append(state_data['action'])
                train_data_z.append(state_data['vars'])
        train_data_x = np.array(train_data_x)
        train_data_y = np.array(train_data_y)
        train_data_z = np.array(train_data_z)
        print(train_data_x.shape, train_data_y.shape, train_data_z.shape)

def cmd_line_args(argv):
    '''
    cmd_line_args() is used to parse command line arguments.
    Returns list of arguments.

    '''

    filename = ''
    try:
        opts, args = getopt.getopt(argv[1:], "ht:")
    except getopt.GetoptError:
        print("Error: VisualDoomAI.py -t <train_data>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("VisualDoomAI.py -t <train_data>")
            sys.exit()
        elif opt in ('-t'):
            filename = arg
    if filename == '':
        print("Error: VisualDoomAI.py -t <train_data>")
        sys.exit(2)
    args = [filename]
    return args

if __name__ == '__main__':
    args = cmd_line_args(sys.argv)
    filename = args[0]

    ai = VisualDoomAI()
    ai.train(filename)
