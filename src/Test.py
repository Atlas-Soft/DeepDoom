#!/usr/bin/python3
'''
Test.py
Authors: Rafael Zamora
Last Updated: 3/3/17

Notes:
-Clipping and Prioritized replay

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
This script is used to run test on trained DQN models, trained Hierarchical-DQN models,
and allow human play to test out scenarios.

"""

# Testing Parameters
scenario = 'configs/rigid_turning.cfg'
model_weights = "best_ddql_rt_.h5"
depth_radius = 1.0
depth_contrast = 1.0
test_param = {
    'frame_skips' : 4,
    'nb_frames' : 3
}
nb_runs = 10
testing = 'DQN'

def test_model(runs=1):
    '''
    Method used to test DQN models on VizDoom scenario. Testing run are replayed
    in higher resolution (800X600).

    Param:

    runs - int : number of test runs done on model.

    '''
    print("Testing DQN-Model:", model_weights)
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Load Model and Weights
    model = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=doom.actions, depth_radius=depth_radius, depth_contrast=depth_contrast)
    model.load_weights(model_weights)
    agent = RLAgent(model, **test_param)

    # Run Scenario and play replay
    for i in range(runs):
        doom = DoomScenario(scenario)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp')

def test_heirarchical_model(runs=1):
    '''
    Method used to test Hierarchical-DQN models on VizDoom scenario. Testing run are replayed
    in higher resolution (800X600).

    Param:

    runs - int : number of test runs done on model.

    '''
    print("Testing Hierarchical-DQN:")
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Load Hierarchical-DQN and Sub-models
    acts = [list(a) for a in it.product([0, 1], repeat=5)]
    actions_1 = []
    actions_2 = []
    for i in range(len(acts)):
        if i < 16: actions_1.append(acts[i])
        if i % 8 == 0: actions_2.append(acts[i])
    model_rigid_turning = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_rigid_turning.load_weights('rigid_turning.h5')
    model_exit_finding = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_exit_finding.load_weights('exit_finding.h5')
    model_doors = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=actions_2, depth_radius=1.0, depth_contrast=0.1)
    model_doors.load_weights('doors.h5')
    models = [model_rigid_turning, model_exit_finding, model_doors]
    model = HDQNModel(sub_models=models, skill_frame_skip=6, resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=[], depth_radius=depth_radius, depth_contrast=depth_contrast)
    model.load_weights(model_weights)
    agent = RLAgent(model, **test_param)

    # Run Scenario and play replay using Hierarchical-DQN
    for i in range(runs):
        doom = DoomScenario(scenario)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp')

def play():
    '''
    Method used to test Vizdoom Scenarios with human players.

    '''
    #Initiates VizDoom Scenario and play
    doom = DoomScenario(scenario)
    doom.apprentice_run()

if __name__ == '__main__':
    if testing == 'DQN': test_model(nb_runs)
    if testing == 'HDQN': test_heirarchical_model(nb_runs)
    if testing == 'human': play()
