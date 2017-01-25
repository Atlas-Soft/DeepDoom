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

def human_play():
    pass

def ai_play():
    sim = DoomSim()
    sim.ai_play()

def replay():
    sim = DoomSim()
    sim.replay("ai_map01_2017-01-25_15:05:14.lmp")

def process_data():
    dp = DataProcessor()
    dp.process_replays()

def train_models():
    pass

if __name__ == '__main__': replay()
