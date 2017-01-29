#!/usr/bin/python3
'''
Visual-Doom-AI: Models.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 1/29/17
CHANGE-LOG:
    1/29/17
        - ADDED Comments

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

class StatePredictionModel():

    def __init__(self):
        '''
        Method initializes the State Prediction Model used to predict future states
        of the Doom environment.

        For more information on this model go to: /doc/models/StatePredictionModel.png

        '''
        #Parameters
        self.optimizer = 'rmsprop'
        self.loss_fun = 'mse'
        self.batch_size = 25
        self.epochs = 100

        #Input Layers
        x0 = Input(shape=(5, 120, 160), name='image_input')
        x1 = Input(shape=(10,), name='action_input')

        #Convolutional Layers
        m = Convolution2D(96, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_1')(x0)
        m = Convolution2D(32, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_2')(m)
        m = Convolution2D(8, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_3')(m)
        m = Flatten()(m)

        #Tranformation Layers
        z= Dense(2400, name='h_layer')(m)
        t = Dense(2400, name='a_layer')(x1)
        m = merge([z, t], mode='mul')

        #Deconvolution Layers
        m = Dense(2400, activation='relu', name='deconv_1')(m)
        m = Reshape((8, 15, 20))(m)
        m = Deconvolution2D(32, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 32, 30, 40), name='deconv_2')(m)
        m = Deconvolution2D(96, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 96, 60, 80), name='deconv_3')(m)
        y0 = Deconvolution2D(1, 5, 5, activation='sigmoid', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 1, 120, 160), name='image_output')(m)

        self.model = Model(input=[x0, x1], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        self.model.summary()

    def load_weights(self, filename):
        '''
        Method loads .h5 weight files from /data/ai_model_weights.

        '''
        self.model.load_weights('../data/ai_model_weights/' + filename)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''
        Method saves .h5 weight files to /data/ai_model_weights.

        '''
        self.model.save_weights('../data/ai_model_weights/' + filename, overwrite=True)

    def train(self, x0, x1, y0):
        '''
        Method trains model using x0, x1, y0 training datasets

        input x0 - (batch_size, 5, 1, 120, 160) tensor; visual data
        input x1 - (batch_size, 9) tensor; action data
        output y0 = (batch_size, 1, 120, 160) tensor; visual data

        '''
        self.model.fit({'image_input': x0, 'action_input': x1}, {'image_output': y0}, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1)

    def test(self, x0, x1, y0):
        '''
        Method used to evaulate model accuracy using x0, x1, y0 validation datasets

        '''
        results = self.model.evaluate({'image_input': x0, 'action_input': x1}, {'image_output': y0}, batch_size=self.batch_size, verbose=1)
        print(results)

    def predict(self, x0, x1):
        '''
        Method used to predict y0 from x0 and x1 datasets

        '''
        return self.model.predict({'image_input': x0, 'action_input': x1}, batch_size=25)

    def prepare_data_sets(self, buffers, actions):
        '''
        Method used to prepare x0, x1, y0 datasets from buffer and action data.

        An element of x0 is a list of 5, (1, 120, 180) buffers from buffer data.
        An element of x1 is an action vector.
        y0 are the buffer data shifted t + 1.
        '''
        x0 = np.delete(buffers, -1, 0)
        x1 = np.delete(actions, -1, 0)
        y0 = np.delete(buffers, 0, 0)

        x0 = np.delete(x0, list(range((x0.shape[0]%self.batch_size))), 0)
        x1 = np.delete(x1, list(range((x1.shape[0]%self.batch_size))), 0)
        y0 = np.delete(y0, list(range((y0.shape[0]%self.batch_size))), 0)

        x0_prime = []
        for i in range(len(x0)):
            temp = []
            for j in range(5):
                k = i - j
                if k > 0: temp.append(x0[k])
                else: temp.append(x0[0])
            x0_prime.append(temp)
        x0 = np.array(x0_prime).reshape(x0.shape[0], 5, 120, 160)

        return x0, x1, y0


class PolicyModel():

    def __init__(self):
        '''

        '''
        #Parameters
        self.optimizer = 'rmsprop'
        self.loss_fun = 'categorical_crossentropy'
        self.batch_size = 25
        self.epochs = 2

        #Input Layers
        x0 = Input(shape=(5, 120, 160), name='image_input')

        #Convolutional Layers
        m = Convolution2D(96, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_1')(x0)
        m = Convolution2D(32, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_2')(m)
        m = Convolution2D(8, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_3')(m)
        m = Flatten()(m)

        #Output Layer
        y0 = Dense(10, activation='softmax', name='action_output')(m)

        self.model = Model(input=[x0,], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        self.model.summary()

    def load_weights(self, filename):
        '''

        '''
        self.model.load_weights('../data/ai_model_weights/' + filename)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''

        '''
        self.model.save_weights('../data/ai_model_weights/' + filename, overwrite=True)

    def train(self, x0, y0):
        '''

        '''
        self.model.fit({'image_input': x0}, {'action_output': y0}, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1)

    def test(self, x0, y0):
        '''

        '''
        results = self.model.evaluate({'image_input': x0}, {'image_output': y0}, batch_size=self.batch_size, verbose=1)
        print(results)

    def predict(self, x0):
        '''

        '''
        return self.model.predict({'image_input': x0}, batch_size=1)

    def prepare_data_sets(self, buffers, actions):
        '''

        '''
        x0 = np.delete(buffers, -1, 0)
        y0 = np.delete(actions, -1, 0)

        x0 = np.delete(x0, list(range((x0.shape[0]%self.batch_size))), 0)
        y0 = np.delete(y0, list(range((y0.shape[0]%self.batch_size))), 0)

        x0_prime = []
        for i in range(len(x0)):
            temp = []
            for j in range(5):
                k = i - j
                if k > 0: temp.append(x0[k])
                else: temp.append(x0[0])
            x0_prime.append(temp)
        x0 = np.array(x0_prime).reshape(x0.shape[0], 5, 120, 160)

        return x0, y0

class StateEvaluationModel():

    def __init__(self):
        '''

        '''
        #Parameters
        self.optimizer = 'rmsprop'
        self.loss_fun = 'binary_crossentropy'
        self.batch_size = 25
        self.epochs = 2

        #Input Layers
        x0 = Input(shape=(1, 120, 160), name='image_input')

        #Convolutional Layer
        m = Convolution2D(96, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_1')(x0)
        m = Convolution2D(32, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_2')(m)
        m = Convolution2D(8, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_3')(m)
        m = Flatten()(m)

        #Output Layer
        y0 = Dense(1, activation='softmax', name='action_output')(m)

        self.model = Model(input=[x0,], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        self.model.summary()

    def load_weights(self, filename):
        '''

        '''
        self.model.load_weights('../data/ai_model_weights/' + filename)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''

        '''
        self.model.save_weights('../data/ai_model_weights/' + filename, overwrite=True)

    def train(self, x0, y0):
        '''

        '''
        self.model.fit({'image_input': x0}, {'action_output': y0}, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1)

    def test(self, x0, y0):
        '''

        '''
        results = self.model.evaluate({'image_input': x0}, {'image_output': y0}, batch_size=self.batch_size, verbose=1)
        print(results)

    def predict(self, x0):
        '''

        '''
        return self.model.predict({'image_input': x0}, batch_size=1)

    def prepare_data_sets(self, buffers, rewards):
        '''

        '''
        x0 = buffers
        y0 = rewards

        x0 = np.delete(x0, list(range((x0.shape[0]%self.batch_size))), 0)
        y0 = np.delete(y0, list(range((y0.shape[0]%self.batch_size))), 0)

        return x0, y0

def MonteCarloModel():

    def __init__(self, spm, pm, sem, actions):
        '''

        '''
        self.state_prediction_model = spm
        self.policy_model = pm
        self.state_evaluation_model = sem
        self.actions = actions
        self.root = None

    def run(self, buffer_, cycles):
        '''

        '''
        self.root = MonteCarloNode(buffer_)
        for i in range(cycles):
            node = self.select()
            node_prime = self.expand(node)
            reward = self.simulate(node_prime)
            self.back_progate(node_prime, reward)
        return self.root.select().action

    def select(self):
        '''

        '''
        node = self.root
        while len(node.childs) != 0: node = node.select()
        return node

    def expand(self, node):
        '''

        '''
        action = self.policy_model.predict(node.buffer_)

        node_prime = MonteCarloNode(self.state_prediction_model.predict(node.buffer_, action), action)
        node_prime.parent = node
        node.childs.append(node_prime)

        return node_prime

    def simulate(self, node):
        '''

        '''
        buffer_ = node.buffer_
        for i in range(100):
            action = self.policy_model.predict(node.buffer_)
            buffer_ = self.state_prediction_model.predict(buffer_, action)

        reward = self.state_evaluation_model.predict(buffer_)
        return reward

    def back_progate(self, node, reward):
        '''

        '''
        while node != self.root:
            node.visits += 1
            node.value += reward
            node = node.parent

def MonteCarloNode():

    def __init__(self, buffer_, action):
        '''

        '''
        self.buffer_ = buffer_
        self.action = action
        self.parent = None
        self.childs = []
        self.value = 0
        self.visits = 1

    def select(self):
        '''

        '''
        node = self.childs[0]
        for i in range(len(self.childs)):
            node_value = node.value
            node_visits = node.visits
            child_value = self.childs[i].value
            child_visits = self.childs[i].visits
            if self.UCT(node_value, node_visits) < self.UCT(child_value, child_visits):
                node = self.childs[i]
        return node

    def UCT(self, value, visits):
        return value * np.sqrt([np.log([self.visits])[0]/visits])[0]
