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
        self.epochs = 10

        #Input Layers
        x0 = Input(shape=(1, 120, 160), name='image_input')
        #a1 = Input(shape=(10,), name='action_input')

        #Convolutional Layers
        m = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='conv_1')(x0)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='conv_2')(m)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Convolution2D(8, 5, 5, border_mode='same', activation='relu', name='conv_3')(m)
        m = MaxPooling2D(pool_size=(2, 2))(m)
        m = Flatten()(m)

        #Variational Encoder Layers
        self.z_mean = Dense(2400, name='z_mean')(m)
        self.z_log_sigma = Dense(2400, name='z_log_sigma')(m)
        z = Lambda(self.vae_sampling, output_shape=(2400,), name='latent_var')([z_mean, z_log_sigma])
        #m = merge([m, mt], mode='concat')

        #Deconvolution Layers
        m = Dense(2400, activation='relu', name='deconv_1')(z)
        m = Reshape((8, 15, 20))(m)
        m = Deconvolution2D(32, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(batch_s, 32, 30, 40), name='deconv_2')(m)
        m = Deconvolution2D(64, 5, 5, activation='relu', border_mode='same', subsample=(2,2), output_shape=(batch_s, 64, 60, 80), name='deconv_3')(m)
        y0 = Deconvolution2D(1, 5, 5, activation='sigmoid', border_mode='same', subsample=(2,2), output_shape=(batch_s, 1, 120, 160), name='image_output')(m)

        self.model = Model(input=[x0,], output=[b0,])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])
        self.model.summary()

    def vae_sampling(self, arg):
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
        kl_loss = 0.5 * K.sum(K.square(z_mean) + K.square(z_log_sigma) - K.log(K.square(z_log_sigma)) - 1, axis=-1)
        return K.mean(gen_loss + kl_loss)

    def load_weights(self):
        '''

        '''
        self.model.load_weights('../data/visual_doom_ai/weights.h5')
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self):
        '''

        '''
        self.model.save_weights('../data/visual_doom_ai/weights.h5', overwrite=True)

    def train(self, x0, y0):
        '''

        '''
        self.model.fit({'image_input': x0}, {'image_output': y0}, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1)

    def test(self, x0, y0):
        pass

    def predict(self, x0):
        '''

        '''
        return self.model.predict({'image_input': x0}, batch_size=self.batch_size)

    def prepare_data_sets(self, buffers, actions):
        '''

        '''
        pass
