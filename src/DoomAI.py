#!/usr/bin/python3
'''
Visual-Doom-AI: DoomAI.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 1/29/17
CHANGE-LOG:
    1/29/17
        - ADDED Comments

'''

"""
Script defines the Doom Ai used in the project.

"""

from random import choice
import numpy as np

class DoomAI():

    def __init__(self, actions):
        '''
        Method initializes Doom Ai variables.

        '''
        self.actions = actions
        np.random.seed(123)

    def act(self, buffer_):
        '''
        Method used to define Doom Ai's behavior.

        ***Currently set to make random action from action list.

        '''
        action = choice(self.actions)
        return action
