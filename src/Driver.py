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
scenario = 'configs/rigid_turning.cfg'
frame_skips = 5
depth_radius = 0.225
depth_contrast = 0.875
qlearn_param = {
    'target_update' : 1000,
    'nb_epoch' : 50,
    'steps' : 5000,
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : 1.0,
    'gamma' : 0.9,
    'epsilon' : [1.0, 0.1],
    'epsilon_rate' : 1.0,
    'observe' : 10,
    'checkpoint' : 5,
    'filename' : 'rigid_turning_.h5'
}

def train():
    '''
    '''

    #Initiates VizDoom Scenario
    doom = Doom(scenario, frame_skips=frame_skips, depth_radius=depth_radius, depth_contrast=depth_contrast)

    # Preform Q Learning on Scenario
    model = DDQModel(resolution=doom.get_state().shape[-2:], nb_frames=qlearn_param['nb_frames'], nb_actions=len(doom.actions))
    agent = QLearnAgent(model, **qlearn_param)
    agent.train(doom)
    model.save_weights("rigid_turning.h5")

def play():
    '''
    Method used to show trained model playing Vizdoom Scenario.

    '''
    #Initiates VizDoom Scenario
    doom = Doom(scenario, frame_skips=4, depth_radius=depth_radius, depth_contrast=depth_contrast)

    # Run Scenario and play replay
    model = DDQModel(resolution=doom.get_state().shape[-2:], nb_frames=qlearn_param['nb_frames'], nb_actions=len(doom.actions))
    model.load_weights('rigid_turning_.h5')
    agent = QLearnAgent(model, **qlearn_param)
    doom.run(agent, save_replay='test.lmp', verbose=True)
    doom.replay('test.lmp')


def test():
    '''
    Method used to test Vizdoom Scenarios with human player.

    '''
    #Initiates VizDoom Scenario and play
    doom = Doom('configs/rigid_turning.cfg', depth_radius=depth_radius, depth_contrast=depth_contrast)
    #plt.imshow(doom.get_state().reshape((120, 160)), interpolation='nearest', cmap='gray')
    #plt.show()
    doom.replay('rigid_turning_.lmp')
    #doom.human_play()

if __name__ == '__main__':
    train()
    #test()
    #for i in range(10): play()
