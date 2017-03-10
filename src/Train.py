#!/usr/bin/python3
'''
Train.py
Authors: Rafael Zamora
Last Updated: 3/3/17

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from RLAgent import RLAgent
from DoomScenario import DoomScenario
from Models import DQNModel, HDQNModel
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

"""
This script is used to train DQN models and Hierarchical-DQN models.

"""

# Training Parameters
scenario = 'configs/all_skills.cfg'
model_weights = 'all_skills.h5'
depth_radius = 1.0
depth_contrast = 0.5
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 6,
    'nb_epoch' : 100,
    'steps' : 5000,
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [1.0, 0.3],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.99,
    'epsilon' : [1.0, 0.1],
    'epsilon_rate' : 0.35,
    'epislon_wait' : 10,
    'nb_tests' : 30,
    'checkpoint' : 1,
    'filename' : 'all_skills_.h5'
}
training = 'HDQN'

def train_model():
    '''
    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Initiates Model
    model = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=doom.actions, depth_radius=depth_radius, depth_contrast=depth_contrast)
    agent = RLAgent(model, **learn_param)

    # Preform Reinforcement Learning on Scenario
    agent.train(doom)
    model.save_weights(model_weights)

def train_heirarchical_model():
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Initiates Hierarchical-DQN model and loads Sub-models
    acts = [list(a) for a in it.product([0, 1], repeat=5)]
    actions_1 = []
    actions_2 = []
    for i in range(len(acts)):
        if i < 16: actions_1.append(acts[i])
        if i % 8 == 0: actions_2.append(acts[i])
    model_rigid_turning = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_rigid_turning.load_weights('rigid_turning.h5')
    model_exit_finding = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_exit_finding.load_weights('exit_finding.h5')
    model_doors = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=actions_2, depth_radius=1.0, depth_contrast=0.1)
    model_doors.load_weights('doors.h5')
    models = [model_rigid_turning, model_exit_finding, model_doors]
    model = HDQNModel(sub_models=models, skill_frame_skip=6, resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], actions=[], depth_radius=depth_radius, depth_contrast=depth_contrast)
    agent = RLAgent(model, **learn_param)

    # Preform Reinforcement Learning on Scenario using Hierarchical-DQN model
    agent.train(doom)
    model.save_weights(model_weights)

if __name__ == '__main__':
    if training == 'DQN': train_model()
    elif training == 'HDQN': train_heirarchical_model()
