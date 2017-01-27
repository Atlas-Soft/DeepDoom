#!/usr/bin/python3
'''
Visual-Doom-AI:
Authors:
Last Updated:
CHANGE-LOG:

'''

"""

"""

from DataProcessor import *
from DoomSim import *
from Models import *
import numpy as np
from matplotlib import pyplot as plt

def human_play():
    '''

    '''
    sim = DoomSim()
    sim.human_play(save=False)
    pass

def ai_play():
    '''

    '''
    sim = DoomSim()
    sim.ai_play()

def replay():
    '''

    '''
    sim = DoomSim()
    sim.replay("player_map01_2016-12-23_11:22:23.lmp")

def process_data():
    '''

    '''
    dp = DataProcessor()
    dp.process_replays()

def train_models():
    '''

    '''
    #State Prediction Model training
    buffers, actions, rewards = load_data("player_map01_2016-12-23_11:22:23.json")
    spm = StatePredictionModel()
    x0, x1, y0 = spm.prepare_data_sets(buffers, actions)
    #spm.load_weights("spm_0_0.h5")
    spm.train(x0, x1, y0)
    spm.save_weights("spm_0_0.h5")
    '''
    #Policy Model training
    buffers, actions, rewards = load_data("player_map01_2016-12-23_11:22:23.json")
    pm = PolicyModel()
    x0, y0 = pm.prepare_data_sets(buffers[:101], actions[:101])
    print(x0.shape, y0.shape)
    #pm.train(x0, y0)
    #pm.save_weights("pm_0_0.h5")

    #State Evaluation Model training
    buffers, actions, rewards = load_data("player_map01_2016-12-23_11:22:23.json")
    sem = StateEvaluationModel()
    x0, y0 = sem.prepare_data_sets(buffers[:101], rewards[:101])
    print(x0.shape, y0.shape)

    #sem.train(x0, y0)
    #sem.save_weights("sem_0_0.h5")
    '''

    x0, x1, y0 = spm.prepare_data_sets(buffers, actions)
    result = spm.predict(x0, x1)

    plt.imshow(result[0].reshape(120,160), interpolation='nearest', cmap='gray')
    plt.figure()
    plt.imshow(y0[0].reshape(120,160), interpolation='nearest', cmap='gray')
    plt.show()

def test_models():
    '''

    '''
    #State Prediction Model evaluation

if __name__ == '__main__':
    '''

    '''
    process_data()
    train_models()
