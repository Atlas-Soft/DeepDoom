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
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import *
from keras.utils import np_utils
from keras import objectives
from keras.models import model_from_json
import keras.backend as K

class VisualDoomAI():

    def __init__(self):
        np.random.seed(123)

    def act(self, buffers, actions):
        action = choice(actions)
        return action

    def prepare_dataset(self, filename):
        '''
        buffers : shape(120, 160, 4), R/G/B/DEPTH int [0,255]
        action : shape(10,), 10 possible action input int [0,1]

        Input - filename of converted data file.
        Output - x0 : shape(len, 1, 120, 160), float [0.0, 1.0]
                 x1 : shape(len, 10)
                 y1 : shape(len, 1, 120, 160), float [0.0, 1.0], x0 but shifted forward by 1
        '''
        x0 = []
        x1 = []
        with open("../data/doom_buffer_action/" + filename, "r") as f:
            for json_dump in f:
                state_data = b64decode_state_data(json_dump)
                grey_buffer_float = np.dot((state_data['buffers'].astype('float32')/255)[...,:3], [0.21, 0.72, 0.07])
                depth_buffer_float =  np.dot((state_data['buffers'].astype('float32')/255)[...,3:4], [1.0,])
                depth_buffer_float[(depth_buffer_float > .25)] = .25
                depth_buffer_filtered = (depth_buffer_float - np.amin(depth_buffer_float))/ (np.amax(depth_buffer_float) - np.amin(depth_buffer_float))
                img_buffer = grey_buffer_float + (.75* (1- depth_buffer_filtered))
                img_buffer_norm = (img_buffer - np.amin(img_buffer))/ (np.amax(img_buffer) - np.amin(img_buffer))
                x0.append(img_buffer_norm)
                x1.append(state_data['action'])
        x0 = np.array(x0)
        x0 = x0.reshape(x0.shape[0], 1, 120, 160)
        y0 = np.delete(x0, 0, 0)
        x0 = np.delete(x0, -1, 0)
        x1 = np.array(x1)
        x1 = np.delete(x1, -1, 0)
        return x0, x1, y0

    def train(self, train_data_filename):
        #Prepare Dataset
        x0, x1, y0 = self.prepare_dataset(train_data_filename)

        batch_s = 25
        x0 = np.delete(x0, list(range((x0.shape[0]%batch_s))), 0)
        x1 = np.delete(x1, list(range((x1.shape[0]%batch_s))), 0)
        y0 = np.delete(y0, list(range((y0.shape[0]%batch_s))), 0)
        print(x0.shape, x1.shape, y0.shape)

        #Initiate Model
        ''''
        a0 = Input(shape=(1, 120, 160), name='image_input')
        a1 = Input(shape=(10,), name='action_input')

        m = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(a0)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Convolution2D(32, 5, 5, border_mode='same', activation='relu' )(m)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Convolution2D(8, 5, 5, border_mode='same', activation='relu' )(m)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Flatten()(m)

        z_mean = Dense(2400)(m)
        z_log_sigma = Dense(2400)(m)
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(25, 2400), mean=0., std=1.0)
            return z_mean + (K.exp(z_log_sigma) * epsilon)
        z = Lambda(sampling, output_shape=(2400,))([z_mean, z_log_sigma])
        #m = merge([m, mt], mode='concat')
        m = Dense(2400, activation='relu')(z)
        m = Reshape((8, 15, 20))(m)

        m = Deconvolution2D(32, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(batch_s, 32, 30, 40))(m)
        m = Deconvolution2D(64, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(batch_s, 64, 60, 80))(m)
        b0 = Deconvolution2D(1, 5, 5, activation='sigmoid', border_mode='same', subsample=(2,2), output_shape=(batch_s, 1, 120, 160), name='image_output')(m)

        model = Model(input=[a0,], output=[b0,])

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = -K.sum(x * K.log(K.epsilon() + x_decoded_mean) + (1-x) * K.log(K.epsilon() + 1 - x_decoded_mean), axis=-1)
            kl_loss = 0.5 * K.sum(K.square(z_mean) + K.square(z_log_sigma) - K.log(K.square(z_log_sigma)) - 1, axis=-1)
            return K.mean(xent_loss + kl_loss)

        #model.load_weights('../data/visual_doom_ai/weights.h5')
        model.compile(optimizer='rmsprop', loss=vae_loss, metrics=['accuracy'])
        model.summary()
        '''
        #model = self.load_model()
        #model.fit({'image_input': x0[:100]}, {'image_output': x0[:100]}, batch_size=batch_s, nb_epoch=5, verbose=1)
        #self.save_model(model)

        #results = model.predict({'image_input': x0[:100]}, batch_size=25)
        #results = results.reshape(results.shape[0], 120, 160)
        #plt.imshow(results[0], cmap='gray')
        #plt.figure()
        plt.imshow(x0[0].reshape(120,160), interpolation='nearest', cmap='gray')
        plt.show()

        import gc; gc.collect()

    def load_model(self):
        model = model_from_json(open('../data/visual_doom_ai/model.json').read())
        model.load_weights('../data/visual_doom_ai/weights.h5')
        model.compile(optimizer=SGD(lr=.0075, nesterov=True), loss='msle', metrics=['accuracy'])
        return model

    def save_model(self, model):
        json_string = model.to_json()
        open('../data/visual_doom_ai/model.json', 'w').write(json_string)
        model.save_weights('../data/visual_doom_ai/weights.h5', overwrite=True)

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
