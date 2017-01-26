#!/usr/bin/python3
'''
Visual-Doom-AI:
Authors:
Last Updated:
CHANGE-LOG:

'''

"""

"""

import keras.backend as K
import numpy as np

from keras.models import Model
from keras.layers import *

class StatePredictionModel():

    def __init__(self):
        '''

        '''
        #Parameters
        self.optimizer = 'rmsprop'
        self.loss_fun = self.vae_loss
        self.batch_size = 25
        self.epochs = 2

        #Input Layers
        x0 = Input(shape=(10, 1, 120, 160), name='image_input')
        x1 = Input(shape=(10,), name='action_input')

        #LSTM Convolutional Layers
        m = ConvLSTM2D(64, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=True, name='conv_1')(x0)
        m = ConvLSTM2D(32, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=True, name='conv_2')(m)
        m = ConvLSTM2D(8, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=False, name='conv_3')(m)
        m = Flatten()(m)

        #Variational Encoder Layers
        self.z_mean = Dense(2400, name='z_mean')(m)
        self.z_log_sigma = Dense(2400, name='z_log_sigma')(m)
        z = Lambda(self.vae_sampling, output_shape=(2400,), name='latent_var')([self.z_mean, self.z_log_sigma])
        m = merge([z, x1], mode='concat')

        #Deconvolution Layers
        m = Dense(2400, activation='relu', name='hidden_1')(m)
        m = Dense(2400, activation='relu', name='deconv_1')(m)
        m = Reshape((8, 15, 20))(m)
        m = Deconvolution2D(32, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 32, 30, 40), name='deconv_2')(m)
        m = Deconvolution2D(64, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 64, 60, 80), name='deconv_3')(m)
        y0 = Deconvolution2D(1, 5, 5, activation='sigmoid', border_mode='same', subsample=(2,2), output_shape=(self.batch_size, 1, 120, 160), name='image_output')(m)

        self.model = Model(input=[x0, x1], output=[y0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        self.model.summary()

    def vae_sampling(self, args):
        '''

        '''
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(25, 2400), mean=0., std=1.0)
        return z_mean + (K.exp(z_log_sigma) * epsilon)

    def vae_loss(self, pred, true):
        '''

        '''
        pred = K.flatten(pred)
        true = K.flatten(true)
        gen_loss = -K.sum(pred * K.log(K.epsilon() + true) + (1-pred) * K.log(K.epsilon() + 1 - true), axis=-1)
        kl_loss = 0.5 * K.sum(K.square(self.z_mean) + K.square(self.z_log_sigma) - K.log(K.square(self.z_log_sigma)) - 1, axis=-1)
        return K.mean(gen_loss + kl_loss)

    def load_weights(self, filename):
        '''

        '''
        self.model.load_weights('../data/ai_model_weights/' + filename)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''

        '''
        self.model.save_weights('../data/ai_model_weights/' + filename, overwrite=True)

    def train(self, x0, x1, y0):
        '''

        '''
        self.model.fit({'image_input': x0, 'action_input': x1}, {'image_output': y0}, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1)

    def test(self, x0, x1, y0):
        '''

        '''
        results = self.model.evaluate({'image_input': x0, 'action_input': x1}, {'image_output': y0}, batch_size=self.batch_size, verbose=1)
        print(results)

    def predict(self, x0, x1):
        '''

        '''
        return self.model.predict({'image_input': x0, 'action_input': x1}, batch_size=1)

    def prepare_data_sets(self, buffers, actions):
        '''

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
            for j in range(10):
                k = i - j
                if k > 0: temp.append(x0[k])
                else: temp.append(x0[0])
            x0_prime.append(temp)
        x0 = np.array(x0_prime)

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
        x0 = Input(shape=(10, 1, 120, 160), name='image_input')

        #LSTM Convolutional Layers
        m = ConvLSTM2D(64, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=True, name='conv_1')(x0)
        m = ConvLSTM2D(32, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=True, name='conv_2')(m)
        m = ConvLSTM2D(8, 5, 5, subsample = (2,2), border_mode='same', activation='relu', return_sequences=False, name='conv_3')(m)
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
            for j in range(10):
                k = i - j
                if k > 0: temp.append(x0[k])
                else: temp.append(x0[0])
            x0_prime.append(temp)
        x0 = np.array(x0_prime)

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
        m = Convolution2D(64, 5, 5, subsample = (2,2), border_mode='same', activation='relu', name='conv_1')(x0)
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
        self.state_prediction_model = spm
        self.policy_model = pm
        self.state_evaluation_model = sem
        self.actions = actions
        self.root = None

    def run(self, buffer_, cycles):
        self.root = MonteCarloNode(buffer_)
        for i in range(cycles):
            node = self.select()
            node_prime = self.expand(node)
            reward = self.simulate(node_prime)
            self.back_progate(node_prime, reward)

        return self.select().action


    def select(self):
        pass

    def expand(self, node):
        pass

    def simulate(self):
        pass

    def back_progate(self, node, reward):
        pass

def MonteCarloNode():

    def __init__(self, buffer_, action):
        self.buffer_ = buffer_
        self.action = action
        self.parent = None
        self.value = 0
        self.visits = 0
