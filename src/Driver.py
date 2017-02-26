#!/usr/bin/python3
'''
Driver.py
Authors: Rafael Zamora
Last Updated: 2/18/17

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from RLAgent import RLAgent
from DoomScenario import DoomScenario
from Models import DModel
import matplotlib.pyplot as plt

# Parameters
scenario = 'configs/exit_finding.cfg'
depth_radius = 1.0
depth_contrast = 0.9
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 6,
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
    'epsilon_rate' : 0.7,
    'epislon_wait' : 10,
    'checkpoint' : 1,
    'filename' : 'exit_finding_.h5'
}

def train():
    '''
    '''
    #Initiates VizDoom Scenario
    doom = DoomScenario(scenario)

    # Preform Q Learning on Scenario
    model = DModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], nb_actions=len(doom.actions), depth_radius=depth_radius, depth_contrast=depth_contrast)
    agent = RLAgent(model, **learn_param)
    agent.train(doom)
    model.save_weights("exit_finding.h5")

def test():
    '''
    Method used to show trained model playing Vizdoom Scenario.

    '''
    #Initiates VizDoomScenario Scenario
    doom = DoomScenario(scenario)
    plt.imshow(doom.get_processed_state(depth_radius, depth_contrast), interpolation='nearest', cmap='gray')
    plt.show()

    # Run Scenario and play replay
    model = DModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], nb_frames=learn_param['nb_frames'], nb_actions=len(doom.actions), depth_radius=depth_radius, depth_contrast=depth_contrast)
    model.load_weights('rigid_turning.h5')
    agent = RLAgent(model, **learn_param)
    for i in range(10):
        doom = DoomScenario(scenario)
        doom.run(agent, save_replay='test.lmp', verbose=True)
        doom.replay('test.lmp')


def play():
    '''
    Method used to test Vizdoom Scenarios with human player.

    '''
    #Initiates VizDoomScenario Scenario and play
    doom = DoomScenario('configs/rigid_turning.cfg', depth_radius=depth_radius, depth_contrast=depth_contrast)
    doom.human_play()

if __name__ == '__main__':
    train()
    #test()
    #play()
