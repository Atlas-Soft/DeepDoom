#!/usr/bin/python3
'''
Models.py
Authors: Rafael Zamora
Last Updated: 3/4/17

'''

"""
Script defines the models used by the Doom Ai.
Models are built using the Keras high-level neural network library.
Keras uses TensorFlow and Theano as back-ends.

"""
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD

class DQNModel:
    """
    DQNModel class is used to define DQN models for the
    Vizdoom environment.

    """

    def __init__(self, resolution=(120, 160), nb_frames=1, actions=[], depth_radius=1.0, depth_contrast=0.8):
        '''
        DQN models have the following network architecture:
        - Input : (# of previous frames, img_width, img_length)
        - ConvLayer: 32 filters, 8x8 filter size, 4x4 stride, rectifier activation
        - ConvLayer: 64 filters, 5x5 filter size, 4x4 stride, rectifier activation
        - FullyConnectedLayer : 4032 nodes with 0.5 dropout rate
        - Output: (# of available actions)

        The loss function is mean-squared error.
        The optimizer is RMSprop with a learning rate of 0.0001

        '''
        # Network Parameters
        self.actions = actions
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input Layers
        x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        # Convolutional Layer
        m = Convolution2D(32, 8, 8, subsample = (4,4), activation='relu')(x0)
        m = Convolution2D(64, 5, 5, subsample = (4,4), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        y0 = Dense(len(self.actions))(m)

        self.online_network = Model(input=x0, output=y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)
        #self.online_network.summary()

    def predict(self, game, q):
        '''
        Method selects predicted action from set of available actions using the
        max-arg q value.

        '''
        a = self.actions[q]
        return a

    def load_weights(self, filename):
        '''
        Method loads DQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.load_weights('../data/model_weights/' + filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''
        Method saves DQN model weights to file located in /data/model_weights/ folder.

        '''
        self.online_network.save_weights('../data/model_weights/' + filename, overwrite=True)

class HDQNModel:
    """
    HDQNModel class is used to define Hierarchical-DQN models for the
    Vizdoom environment.

    Hierarchical-DQN models can be set to use only submodels by setting by not
    passing list of available native actions.

    skill_frames_skip allows for mulitple consecutive uses of sub-model predictions
    if they are chosen by the Hierarchical model. This may help increase effectiveness
    of models which require a longer set of actions to reach any substantial reward.

    """

    def __init__(self, sub_models=[], skill_frame_skip=0, resolution=(120, 160), nb_frames=1, actions=[], depth_radius=1.0, depth_contrast=0.8):
        '''
        Hierarchical-DQN models have the following network architecture:
        - Input : (# of previous frames, img_width, img_length)
        - ConvLayer: 32 filters, 8x8 filter size, 4x4 stride, rectifier activation
        - ConvLayer: 64 filters, 5x5 filter size, 4x4 stride, rectifier activation
        - FullyConnectedLayer : 4032 nodes with 0.5 dropout rate
        - Output: (# of available actions)

        The loss function is mean-squared error.
        The optimizer is RMSprop with a learning rate of 0.0001

        '''
        self.actions = actions
        self.sub_models = sub_models
        self.sub_model_frames = None
        self.nb_frames = nb_frames
        self.nb_actions = len(actions)
        self.last_q = None
        self.skill_frame_skip = skill_frame_skip
        self.skip_count = 0

        # Network Parameters
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input Layers
        x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        # Convolutional Layer
        m = Convolution2D(32, 8, 8, subsample = (4,4), activation='relu', init='normal')(x0)
        m = Convolution2D(64, 5, 5, subsample = (4,4), activation='relu', init='normal')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu', init='normal')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        y0 = Dense(len(self.actions) + len(self.sub_models), init='normal')(m)

        self.online_network = Model(input=x0, output=y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)
        #self.online_network.summary()

    def predict(self, game, q):
        '''
        Method selects predicted action from set of available actions using the
        max-arg q value.

        '''
        # Keep track of sub-model frames for predictions
        # Each sub-model requires their own specifically processed frames.
        if self.sub_model_frames == None:
            temp = []
            for model in self.sub_models:
                frame = game.get_processed_state(model.depth_radius, model.depth_contrast)
                frames = [frame] * self.nb_frames
                temp.append(frames)
            self.sub_model_frames = temp
        else:
            for i in range(len(self.sub_models)):
                model = self.sub_models[i]
                frame = game.get_processed_state(model.depth_radius, model.depth_contrast)
                self.sub_model_frames[i].append(frame)
                self.sub_model_frames[i].pop(0)

        # Get predicted action from sub-models or native actions.
        if self.last_q == None:
            if q >= self.nb_actions:
                q = q - self.nb_actions
                sel_model = self.sub_models[q]
                S = np.expand_dims(self.sub_model_frames[q], 0)
                sel_model_q = sel_model.online_network.predict(S)
                sel_model_q = int(np.argmax(sel_model_q[0]))
                a = sel_model.predict(game, sel_model_q)
                self.last_q = q
            else:
                a = self.actions[q]
        else:
            sel_model = self.sub_models[self.last_q]
            S = np.expand_dims(self.sub_model_frames[self.last_q], 0)
            sel_model_q = sel_model.online_network.predict(S)
            sel_model_q = int(np.argmax(sel_model_q[0]))
            a = sel_model.predict(game, sel_model_q)
            if self.skip_count < self.skill_frame_skip: self.skip_count += 1
            else:
                self.skip_count = 0
                self.last_q = None
        return a

    def load_weights(self, filename):
        '''
        Method loads HDQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.load_weights('../data/model_weights/' + filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=['accuracy'])

    def save_weights(self, filename):
        '''
        Method saves HDQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.save_weights('../data/model_weights/' + filename, overwrite=True)
