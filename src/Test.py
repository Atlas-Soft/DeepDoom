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
from Models import DQNModel, HDQNModel, StatePredictionModel, all_skills_HDQN, all_skills_shooting_HDQN

"""
This script is used to run test on trained DQN models, trained Hierarchical-DQN models,
and allow human play to test out scenarios.

"""

# Testing Parameters
scenario = 'all_skills_shooting.cfg'
model_weights = "distilled_HDQNModel_all_skills.h5"
depth_radius = 1.0
depth_contrast = 0.5
test_param = {
    'frame_skips' : 4,
    'nb_frames' : 3
}
nb_runs = 5
testing = 'HDQN'

def test_model(runs=1):
    '''
    Method used to test DQN models on VizDoom scenario. Testing run are replayed
    in higher resolution (800X600).

    Param:

    runs - int : number of test runs done on model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Load Model and Weights
    model = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=doom.actions, depth_radius=depth_radius, depth_contrast=depth_contrast)
    model.load_weights(model_weights)
    agent = RLAgent(model, **test_param)

    print("\nTesting DQN-Model:", model_weights)
    # Run Scenario and play replay
    for i in range(runs):
        doom = DoomScenario(scenario)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp', doom_like=True)


def test_heirarchical_model(runs=1):
    '''
    Method used to test Hierarchical-DQN models on VizDoom scenario. Testing run are replayed
    in higher resolution (800X600).

    Param:

    runs - int : number of test runs done on model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)
    resolution = doom.get_processed_state(depth_radius, depth_contrast).shape[-2:]

    # Load Hierarchical-DQN and Sub-models
    model = all_skills_shooting_HDQN(resolution, 4, depth_radius, depth_contrast, test_param)
    #model.load_weights(model_weights)
    agent = RLAgent(model, **test_param)

    # Run Scenario and play replay using Hierarchical-DQN
    print("\nTesting Hierarchical-DQN:", model_weights)
    for i in range(runs):
        doom = DoomScenario(scenario)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp', doom_like=True)

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
