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
from keras.layers.advanced_activations import LeakyReLU

class PolicyModel():

    def __init__(self):
        '''

        '''
        #Parameters
        self.nb_actions = 8
        self.optimizer = 'sgd'
        self.loss_fun = 'mse'

        #Input Layers
        x0 = Input(shape=(4, 120, 160))

        #Convolutional Layers
        m = Convolution2D(32, 3, 7, subsample = (1,1), activation='relu')(x0)
        m = Convolution2D(32, 4, 4, subsample = (2,2), activation='relu')(m)
        m = Flatten()(m)

        #Output Layer
        m = Dense(800, activation=LeakyReLU())(m)
        y0 = Dense(8)(m)

        self.model = Model(input=[x0,], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        #self.model.summary()

    def predict(self, S, q):
        '''
        '''
        a = q
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
