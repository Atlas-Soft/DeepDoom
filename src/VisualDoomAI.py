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
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D
from keras.models import model_from_json
import numpy as np

class VisualDoomAI():

    def __init__(self):
        self.doom_model = None
        self.seed = 123456
        try:
            with open("../data/visual_doom_ai/doom_model.json", "r") as f:
                self.doom_model = model_from_json(f.read())
            self.doom_model.load_weights("../data/visual_doom_ai/doom_model.h5")
            self.doom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        except:
            self.init_model()

    def init_model(self):
        np.random.seed(self.seed)
        n_filters = 32
        n_conv = 3
        n_pool = 2
        self.doom_model = Sequential()
        self.doom_model.add(Convolution2D(n_filters, n_conv, n_conv, border_mode='valid',input_shape=(120,160,3)))
        self.doom_model.add(Activation('relu'))
        self.doom_model.add(Convolution2D(n_filters, n_conv, n_conv))
        self.doom_model.add(Activation('relu'))
        self.doom_model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
        self.doom_model.add(Dropout(0.25))
        self.doom_model.add(Flatten())
        self.doom_model.add(Dense(128))
        self.doom_model.add(Activation('relu'))
        self.doom_model.add(Dropout(0.5))
        self.doom_model.add(Dense(13))
        self.doom_model.add(Activation('softmax'))
        self.doom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def act(self, screen_buffer, depth_buffer, actions):
        #i = np.array([np.transpose(screen_buffer, (1, 2, 0))])
        #predictions = self.doom_model.predict(i)[0]
        #rounded = [int(round(x)) for x in predictions]
        action = choice(actions)
        return action

    def train(self, train_data_filename):
        train_data_x = []
        train_data_y = []
        with open("../data/doom_buffer_action/" + train_data_filename, "r") as f:
            for json_dump in f:
                state_data = b64decode_state_data(json_dump)
                train_data_x.append(np.transpose(state_data['screen_buffer'], (1, 2, 0)))
                train_data_y.append(state_data['action'])
        train_data_x = np.array(train_data_x)
        train_data_y = np.array(train_data_y)
        self.doom_model.fit(train_data_x, train_data_y, nb_epoch=1, batch_size=10, verbose=1)

        doom_model_json = self.doom_model.to_json()
        with open("../data/visual_doom_ai/doom_model.json", "w") as json_file:
            json_file.write(doom_model_json)
        self.doom_model.save_weights("../data/visual_doom_ai/doom_model.h5")
        print("Training Done.")

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
