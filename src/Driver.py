#!/usr/bin/python3
'''
Visual-Doom-AI: Driver.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 1/29/17
CHANGE-LOG:
    1/29/17
        - ADDED Comments

'''

"""
Script used as the driver for the project.

Note: Different use-cases should be seperated by method.

"""

from DataProcessor import *
from DoomSim import *
from Models import *
import numpy as np
from matplotlib import pyplot as plt

def human_play():
    '''
    Method used to run a human-played simulation of Vizdoom.

    '''
    sim = DoomSim()
    sim.human_play(save=False)
    pass

def ai_play():
    '''
    Method used to run an ai-played simulation of Vizdoom.

    '''
    sim = DoomSim()
    sim.ai_play()

def replay():
    '''
    Method used to run a replay simulation of Vizdoom.

    '''

    sim = DoomSim()
    sim.replay("player_map01_2016-12-23_11:22:23.lmp")

def process_data():
    '''
    Method used to process replay data in /data/doom_replay_data/.
    Processed data saved in /data/doom_processed_data

    '''
    dp = DataProcessor()
    dp.process_replays()

def train_models():
    '''
    Method used to train models of ai.

    ***Currently set to train State Prediction Model.

    '''
    #State Prediction Model training
    buffers, actions, rewards = load_data("player_map01_2016-12-23_11:22:23.json")
    spm = StatePredictionModel()
    x0, x1, y0 = spm.prepare_data_sets(buffers, actions)
    #spm.load_weights("spm_0_0.h5")
    spm.train(x0, x1, y0)
    spm.save_weights("spm_0_0.h5")

    result = spm.predict(x0, x1)
    plt.imshow(result[0].reshape(120,160), interpolation='nearest', cmap='gray')
    plt.figure()
    plt.imshow(y0[0].reshape(120,160), interpolation='nearest', cmap='gray')
    plt.show()

def test_models():
    '''
    Method used to evaluate accuracy of models.

    '''
    pass

if __name__ == '__main__':
    '''
    ***Currently set to process replay data and train models.

    '''
    process_data()
    train_models()
