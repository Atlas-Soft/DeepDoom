#!/usr/bin/python3
'''
Visual-Doom-AI: Models.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 2/14/17
CHANGE-LOG:

'''

"""
Script defines the models used by the Doom Ai.
Models are built using the Keras high-level neural network library.
Keras uses TensorFlow and Theano as back-ends.

***Current models are in the prototyping phase,

"""

import keras.backend as K
import numpy as np

from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop

class PolicyModel():

    def __init__(self):
        '''

        '''
        #Parameters
        self.optimizer = RMSprop(lr=0.001)
        self.loss_fun = 'categorical_crossentropy'

        #Input Layers
        x0 = Input(shape=(1, 120, 160), name='image_input')

        #Convolutional Layers
        m = Convolution2D(32, 8, 8, subsample = (2,2), border_mode='same', activation='relu', name='conv_1')(x0)
        m = Convolution2D(64, 6, 6, subsample = (2,2), border_mode='same', activation='relu', name='conv_2')(m)
        m = Convolution2D(64, 6, 6, subsample = (2,2), border_mode='same', activation='relu', name='conv_3')(m)
        m = Convolution2D(64, 4, 4, subsample = (2,2), border_mode='same', activation='relu', name='conv_4')(m)
        m = Flatten()(m)

        #Output Layer
        m = Dense(2048, name='h_layer')(m)
        y0 = Dense(8, activation='softmax', name='action_output')(m)

        self.model = Model(input=[x0,], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        #self.model.summary()

    def predict(self, S):
        '''
        '''
        q = self.model.predict(S)
        a = int(np.argmax(q[0]))
        return a

    def load_weights(self, filename):
        '''

        '''
        self.model.load_weights('../data/model_weights/' + filename)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''

        '''
        self.model.save_weights('../data/model_weights/' + filename, overwrite=True)
