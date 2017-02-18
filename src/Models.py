#!/usr/bin/python3
'''
Models.py
Authors: Rafael Zamora
Last Updated: 2/18/17

'''

"""
Script defines the models used by the Doom Ai.
Models are built using the Keras high-level neural network library.
Keras uses TensorFlow and Theano as back-ends.

***Current models are in the prototyping phase,

"""
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD

class QModel():
    """
    Model class used to interface with QLearnAgent

    """

    def __init__(self): pass

    def predict(self, S, q): pass

    def load_weights(self, filename): pass

    def save_weights(self, filename): pass

class DoomQModel(QModel):
    """
    QModel class is used to define Deep Q-Learning models for the Vizdoom

    """

    def __init__(self, resolution=(120, 160), nb_frames=1, nb_actions=2):
        '''

        '''
        #Parameters
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.00025)
        
        #Input Layers
        x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        #Convolutional Layer4
        m = Convolution2D(32, 8, 8, subsample = (4,4), activation='relu')(x0)
        m = Convolution2D(64, 5, 5, subsample = (2,2), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(800, activation='relu')(m)

        #Output Layer
        y0 = Dense(8)(m)

        self.q_net = Model(input=x0, output=y0)
        self.q_net.compile(optimizer=self.optimizer, loss=self.loss_fun)
        #self.q_net.summary()

    def predict(self, S, q):
        '''
        '''
        a = q
        return a

    def load_weights(self, filename):
        '''
        '''
        self.q_net.load_weights('../data/model_weights/' + filename)
        self.q_net.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''
        '''
        self.q_net.save_weights('../data/model_weights/' + filename, overwrite=True)
