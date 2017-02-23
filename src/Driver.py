#!/usr/bin/python3
'''
Driver.py
Authors: Rafael Zamora
Last Updated: 2/18/17

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Qlearning4k import QLearnAgent
from Doom import Doom
from Models import DQModel, DDQModel
import matplotlib.pyplot as plt

# Parameters
scenario = 'configs/exit_finding.cfg'
frame_skips = 6
depth_radius = 1.0
depth_contrast = 0.8
qlearn_param = {
    'target_update' : 0,
    'nb_epoch' : 100,
    'steps' : 5000,
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : 0.1,
    'gamma' : 0.9,
    'epsilon' : [1.0, 0.1],
    'epsilon_rate' : 0.2,
    'observe' : 20,
    'checkpoint' : 1,
    'filename' : 'exit_finding_.h5'
}

def train():
    '''
    '''
    #Initiates VizDoom Scenario
    doom = Doom(scenario, frame_skips=frame_skips, depth_radius=depth_radius, depth_contrast=depth_contrast)

    # Preform Q Learning on Scenario
    model = DQModel(resolution=doom.get_state().shape[-2:], nb_frames=qlearn_param['nb_frames'], nb_actions=len(doom.actions))
    agent = QLearnAgent(model, **qlearn_param)
    agent.train(doom)
    model.save_weights("exit_finding.h5")

def test():
    '''
    Method used to show trained model playing Vizdoom Scenario.

    '''
    #Initiates VizDoom Scenario
    doom = Doom(scenario, frame_skips=6, depth_radius=depth_radius, depth_contrast=depth_contrast)
    plt.imshow(doom.get_state().reshape((120, 160)), interpolation='nearest', cmap='gray')
    plt.show()
    # Run Scenario and play replay
    model = DQModel(resolution=doom.get_state().shape[-2:], nb_frames=qlearn_param['nb_frames'], nb_actions=len(doom.actions))
    model.load_weights('rigid_turning_.h5')
    agent = QLearnAgent(model, **qlearn_param)
    for i in range(5):
        doom = Doom(scenario, frame_skips=6, depth_radius=depth_radius, depth_contrast=depth_contrast)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp')


def play():
    '''
    Method used to test Vizdoom Scenarios with human player.

    '''
    #Initiates VizDoom Scenario and play
    doom = Doom('configs/rigid_turning.cfg', depth_radius=depth_radius, depth_contrast=depth_contrast)
    doom.replay('rigid_turning_.lmp')
    #doom.human_play()

if __name__ == '__main__':
    #train()
    test()
    #play()
