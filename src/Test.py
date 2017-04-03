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
from Models import DQNModel, HDQNModel, StatePredictionModel
import itertools as it
import matplotlib.pyplot as plt
"""
This script is used to run test on trained DQN models, trained Hierarchical-DQN models,
and allow human play to test out scenarios.

"""

# Testing Parameters
scenario = 'all_skills.cfg'
model_weights = "double_dqlearn_HDQNModel_all_skills.h5"
depth_radius = 1.0
depth_contrast = 0.5
test_param = {
    'frame_skips' : 4,
    'nb_frames' : 3
}
nb_runs = 5
testing = 'HDQN'
test_state_prediction = False

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
        doom.replay('test.lmp')

    if test_state_prediction:
        doom = DoomScenario(scenario)
        data = doom.run(agent, return_data=True)
        state_predictor = StatePredictionModel(resolution=model.resolution, nb_frames=model.nb_frames, nb_actions=model.nb_actions, depth_radius=model.depth_radius, depth_contrast=model.depth_contrast)
        state_predictor.load_weights('sp_' + model_weights)
        pred = state_predictor.autoencoder_network.predict(data)
        pred = list(pred)
        for i in range(len(pred)):
            if i == 10: plt.close("all")
            fig = plt.imshow(pred[i][0], cmap='gray', interpolation='nearest')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig("../doc/figures/sp/pred_" + str(i) + ".png", bbox_inches='tight', pad_inches = 0)
            plt.figure()
        for i in range(len(pred)):
            if i == 10: plt.close("all")
            fig = plt.imshow(data[0][i][-1], cmap='gray', interpolation='nearest')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig("../doc/figures/sp/true_" + str(i) + ".png", bbox_inches='tight', pad_inches = 0)
            plt.figure()


def test_heirarchical_model(runs=1):
    '''
    Method used to test Hierarchical-DQN models on VizDoom scenario. Testing run are replayed
    in higher resolution (800X600).

    Param:

    runs - int : number of test runs done on model.

    '''
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
    model_rigid_turning.load_weights('double_dqlearn_DQNModel_rigid_turning.h5')
    model_exit_finding = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=actions_1, depth_radius=1.0, depth_contrast=0.9)
    model_exit_finding.load_weights('double_dqlearn_DQNModel_exit_finding.h5')
    model_doors = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=actions_2, depth_radius=1.0, depth_contrast=0.1)
    model_doors.load_weights('double_dqlearn_DQNModel_doors.h5')
    models = [model_rigid_turning, model_exit_finding, model_doors]
    model = HDQNModel(sub_models=models, skill_frame_skip=4, resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=test_param['nb_frames'], actions=[], depth_radius=depth_radius, depth_contrast=depth_contrast)
    model.load_weights(model_weights)
    agent = RLAgent(model, **test_param)

    # Run Scenario and play replay using Hierarchical-DQN
    print("\nTesting Hierarchical-DQN:", model_weights)
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
