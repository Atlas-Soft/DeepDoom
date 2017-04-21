#!/usr/bin/python3
'''
Train.py
Authors: Rafael Zamora
Last Updated: 3/3/17

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from RLAgent import RLAgent
from DoomScenario import DoomScenario
from Models import DQNModel, HDQNModel, all_skills_HDQN, all_skills_shooting_HDQN
import keras.backend as K
import numpy as np

"""
This script is used to train DQN models and Hierarchical-DQN models.

"""

# Training Parameters
scenario = 'all_skills_shooting.cfg'
model_weights = None
depth_radius = 1.0
depth_contrast = 0.5
learn_param = {
    'learn_algo' : 'double_dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 4,
    'nb_epoch' : 100,
    'steps' : 5000,
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [1.0, 0.1],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.9,
    'epsilon' : [1.0, 0.1],
    'epsilon_rate' : 0.35,
    'epislon_wait' : 10,
    'nb_tests' : 20,
}
training = 'HDQN'
training_arg = [4,'all_skills_shooting']


def train_model():
    '''
    Method trains primitive DQN-Model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Initiates Model
    model = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=doom.actions, depth_radius=depth_radius, depth_contrast=depth_contrast)
    if model_weights: model.load_weights(model_weights)
    agent = RLAgent(model, **learn_param)

    # Preform Reinforcement Learning on Scenario
    agent.train(doom)

def train_heirarchical_model():
    '''
    Method trains Hierarchical-DQN model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)
    resolution = doom.get_processed_state(depth_radius, depth_contrast).shape[-2:]

    # Initiates Hierarchical-DQN model and loads Sub-models
    if training_arg[1] == 'all_skills_shooting':
        model = all_skills_shooting_HDQN(resolution, training_arg[0], depth_radius, depth_contrast, learn_param)
    else:
        model = all_skills_HDQN(resolution, training_arg[0], depth_radius, depth_contrast, learn_param)
    if model_weights: model.load_weights(model_weights)
    agent = RLAgent(model, **learn_param)

    # Preform Reinforcement Learning on Scenario using Hierarchical-DQN model
    agent.train(doom)

def train_distilled_model():
    '''
    Method trains distlled DQN-Model from Hierarchical-DQN model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)
    resolution = doom.get_processed_state(depth_radius, depth_contrast).shape[-2:]

    # Load Hierarchical-DQN and Sub-models
    teacher_model = all_skills_HDQN(resolution, training_arg[0], depth_radius, depth_contrast, learn_param)
    teacher_model.load_weights('double_dqlearn_HDQNModel_all_skills.h5')
    teacher_agent = RLAgent(teacher_model, **learn_param)

    # Initiate Distilled Model
    student_model = DQNModel(distilled=True, resolution=resolution, nb_frames=learn_param['nb_frames'], actions=doom.actions, depth_radius=depth_radius, depth_contrast=depth_contrast)
    student_model.online_network.compile(optimizer='adadelta', loss='kullback_leibler_divergence')
    student_agent = RLAgent(student_model, **learn_param)

    # Preform Transfer Learning on Scenario by distilling Hierarchical-DQN model
    teacher_agent.transfer_train(student_agent, doom)

if __name__ == '__main__':
    if training == 'DQN': train_model()
    elif training == 'HDQN': train_heirarchical_model()
    elif training == 'Distilled-HDQN': train_distilled_model()
    import gc; gc.collect()
