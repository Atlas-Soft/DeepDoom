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
import itertools as it
import numpy as np
np.set_printoptions(precision=3)
import keras.callbacks as KC
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD
from keras.utils import plot_model
from sklearn.preprocessing import normalize

class DQNModel:
    """
    DQNModel class is used to define DQN models for the
    Vizdoom environment.

    """

    def __init__(self, resolution=(120, 160), nb_frames=1, actions=[], depth_radius=1.0, depth_contrast=0.8, distilled=False):
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
        self.resolution = resolution
        self.actions = actions
        self.nb_actions = len(actions)
        self.nb_frames = nb_frames
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input Layers
        self.x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        # Convolutional Layer
        m = Conv2D(32, (8, 8), strides = (4,4), activation='relu', )(self.x0)
        m = Conv2D(64, (5, 5), strides = (4,4), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        if distilled:
            self.y0 = Dense(self.nb_actions, activation='softmax')(m)
        else:
            self.y0 = Dense(self.nb_actions)(m)

        self.online_network = Model(inputs=self.x0, outputs=self.y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)
        self.target_network = None
        self.state_predictor = None
        #self.online_network.summary()
        #plot_model(self.online_network, to_file='../doc/model.png', show_shapes=True, show_layer_names=False)
        #tbcall = KC.TensorBoard(log_dir="../doc/logs", histogram_freq=0, write_graph=True, write_images=True)
        #tbcall.set_model(self)

    def predict(self, game, q):
        '''
        Method selects predicted action from set of available actions using the
        max-arg q value.

        '''
        a = self.actions[q]
        return a

    def softmax_q_values(self, S, actions, q_=None):
        '''
        Method returns softmax of predicted q values indexed according to the
        desired list of actions.

        '''
        # Calculate Softmax of Q values
        q = self.online_network.predict(S)
        max_q = int(np.argmax(q[0]))

        # Index Q values according to inputed list of actions
        final_q = [0 for i in range(len(actions))]
        for j in range(len(model_actions)):
            for i in range(len(actions)):
                if model_actions[j] == actions[i]:
                    final_q[i] = q[0][j]

        # ASk dr. Pierce about sharpening data points.
        final_q = np.array(final_q)
        softmax_q = np.exp((final_q)/0.15)
        softmax_q =  softmax_q / softmax_q.sum(axis=0)

        return softmax_q, max_q

    def load_weights(self, filename):
        '''
        Method loads DQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.load_weights('../data/model_weights/' + filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def save_weights(self, filename):
        '''
        Method saves DQN model weights to file located in /data/model_weights/ folder.

        '''
        self.online_network.save_weights('../data/model_weights/' + filename, overwrite=True)

class HDQNModel:
    """
    HDQNModel class is used to define Hierarchical-DQN models for the
    Vizdoom environment.

    Hierarchical-DQN models can be set to use only submodels by not
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
        self.resolution = resolution
        self.actions = actions
        self.sub_models = sub_models
        self.sub_model_frames = None
        self.nb_frames = nb_frames
        self.nb_actions = len(self.actions) + len(self.sub_models)
        self.skill_frame_skip = skill_frame_skip

        # Network Parameters
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input Layers
        self.x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        # Convolutional Layer
        m = Conv2D(32, (8, 8), strides = (4,4), activation='relu')(self.x0)
        m = Conv2D(64, (5, 5), strides = (4,4), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        self.y0 = Dense(self.nb_actions)(m)

        self.online_network = Model(inputs=self.x0, outputs=self.y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)
        self.target_network = None
        self.state_predictor = None
        #self.online_network.summary()
        #tbcall = KC.TensorBoard(log_dir="../doc/logs", histogram_freq=0, write_graph=True, write_images=True)
        #tbcall.set_model(self)

    def update_submodel_frames(self, game):
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

    def predict(self, game, q):
        '''
        Method selects predicted action from set of available actions using the
        max-arg q value.

        '''
        self.update_submodel_frames(game)

        # Get predicted action from sub-models or native actions.
        if q >= len(self.actions):
            q = q - len(self.actions)
            sel_model = self.sub_models[q]
            S = np.expand_dims(self.sub_model_frames[q], 0)
            sel_model_q = sel_model.online_network.predict(S)
            sel_model_q = int(np.argmax(sel_model_q[0]))
            a = sel_model.predict(game, sel_model_q)
        else:
            a = self.actions[q]
        return a

    def softmax_q_values(self, S, actions, q_=None):
        '''
        Method returns softmax of predicted q values indexed according to the
        desired list of actions.

        '''
        # Calculate Softmax of Q values from Selected DQN
        q = self.online_network.predict(S)
        max_q = int(np.argmax(q[0]))
        if q_: max_q = q_
        if max_q >= len(self.actions):
            max_q = max_q - len(self.actions)
            sel_model = self.sub_models[max_q]
            S = np.expand_dims(self.sub_model_frames[max_q], 0)
            q = sel_model.online_network.predict(S)
            model_actions = sel_model.actions
        else:
            model_actions = self.actions
            q = q[:len(self.actions)]

        #q = normalize(q, norm='max')
        # Index Q values according to inputed list of actions
        final_q = [0 for i in range(len(actions))]
        for j in range(len(model_actions)):
            for i in range(len(actions)):
                if model_actions[j] == actions[i]:
                    final_q[i] = q[0][j]

        # ASk dr. Pierce about sharpening data points.
        # Sharpen q values using Softmax
        final_q = np.array(final_q)
        softmax_q = np.exp((final_q)/1.0)
        softmax_q =  softmax_q / softmax_q.sum(axis=0)
        #print(softmax_q)

        return softmax_q, max_q

    def load_weights(self, filename):
        '''
        Method loads HDQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.load_weights('../data/model_weights/' + filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def save_weights(self, filename):
        '''
        Method saves HDQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.save_weights('../data/model_weights/' + filename, overwrite=True)

class StatePredictionModel:

    def __init__(self, resolution=(120, 160), nb_frames=1, nb_actions=0, depth_radius=1.0, depth_contrast=0.8):
        '''
        Method initializes the State Prediction Model used to predict future states
        of the Doom environment.
        '''
        #Parameters
        self.resolution = resolution
        self.nb_actions = nb_actions
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast
        self.optimizer = RMSprop(lr=0.0005)
        self.loss_fun = 'mse'

        #Input Layers
        x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))
        x1 = Input(shape=(self.nb_actions,))

        #Convolutional Layers
        m = Conv2D(16, (8, 8), strides=(2,2), padding='same', activation='relu')(x0)
        m = BatchNormalization()(m)
        m = Conv2D(32, (6, 6), strides=(2,2), padding='same', activation='relu')(m)
        m = BatchNormalization()(m)
        m = Conv2D(32, (6, 6), strides=(3,2), padding='same', activation='relu')(m)
        m = BatchNormalization()(m)
        m = Conv2D(32, (4, 4), strides=(2,2), padding='same', activation='relu')(m)
        m = BatchNormalization()(m)
        m = Flatten()(m)

        #Tranformation Layers
        z = Dense(1600)(m)
        t = Dense(1600)(x1)
        m = merge([z, t], mode='mul')

        #Deconvolution Layers
        m = Dense(1600, activation='relu')(m)
        m = Reshape((32, 5, 10))(m)
        m = Conv2DTranspose(32, (4, 4), activation='relu', padding='same', strides=(2,2), data_format="channels_first")(m)
        m = BatchNormalization()(m)
        m = Conv2DTranspose(32, (6, 6), activation='relu', padding='same', strides=(3,2), data_format="channels_first")(m)
        m = BatchNormalization()(m)
        m = Conv2DTranspose(16, (6, 6), activation='relu', padding='same', strides=(2,2), data_format="channels_first")(m)
        m = BatchNormalization()(m)
        y0 = Conv2DTranspose(1, (8, 8), padding='same', strides=(2,2), data_format="channels_first")(m)

        self.autoencoder_network = Model(input=[x0, x1], output=[y0,])
        self.autoencoder_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

        tbcall = KC.TensorBoard(log_dir="../doc/logs", histogram_freq=0, write_graph=True, write_images=True)
        tbcall.set_model(self)

    def load_weights(self, filename):
        '''
        Method loads .h5 weight files from /data/ai_model_weights.
        '''
        self.autoencoder_network.load_weights('../data/model_weights/' + filename)
        self.autoencoder_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def save_weights(self, filename):
        '''
        Method saves .h5 weight files to /data/ai_model_weights.
        '''
        self.autoencoder_network.save_weights('../data/model_weights/' + filename, overwrite=True)

def all_skills_HDQN(resolution, skill_frame_skips, depth_radius, depth_contrast, param):
    '''
    '''
    acts = [list(a) for a in it.product([0, 1], repeat=5)]
    actions_1 = []
    actions_2 = []
    for i in range(len(acts)):
        if i < 16: actions_1.append(acts[i])
        if acts[i][2] != 1: actions_2.append(acts[i])
    model_rigid_turning = DQNModel(resolution=resolution, nb_frames=param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_rigid_turning.load_weights('double_dqlearn_DQNModel_rigid_turning.h5')
    model_exit_finding = DQNModel(resolution=resolution, nb_frames=param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_exit_finding.load_weights('double_dqlearn_DQNModel_exit_finding.h5')
    model_doors = DQNModel(resolution=resolution, nb_frames=param['nb_frames'], actions=actions_2, depth_radius=1.0, depth_contrast=0.5)
    model_doors.load_weights('double_dqlearn_DQNModel_doors.h5')
    models = [model_rigid_turning, model_exit_finding, model_doors]
    model = HDQNModel(sub_models=models, skill_frame_skip=skill_frame_skips, resolution=resolution, nb_frames=param['nb_frames'], actions=[], depth_radius=depth_radius, depth_contrast=depth_contrast)
    return model

def all_skills_shooting_HDQN(resolution, skill_frame_skips, depth_radius, depth_contrast, param):
    '''
    '''
    acts = [list(a) for a in it.product([0, 1], repeat=6)]
    actions_1 = []
    actions_2 = []
    for i in range(len(acts)):
        if acts[i][0] != 1: actions_1.append(acts[i])
        if acts[i][1] != 1 and acts[i][2] != 1 and acts[i][3] != 1: actions_2.append(acts[i])
    model_all_skills = DQNModel(resolution=resolution, nb_frames=param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.5)
    model_all_skills.load_weights('distilled_HDQNModel_all_skills_ui.h5')
    model_shooting = DQNModel(resolution=resolution, nb_frames=param['nb_frames'], actions=actions_2, depth_radius=1.0, depth_contrast=0.75)
    model_shooting.load_weights('double_dqlearn_DQNModel_shooting.h5')
    models = [model_shooting, model_all_skills]
    model = HDQNModel(sub_models=models, skill_frame_skip=skill_frame_skips, resolution=resolution, nb_frames=param['nb_frames'], actions=[], depth_radius=depth_radius, depth_contrast=depth_contrast)
    return model
