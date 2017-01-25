#!/usr/bin/python3
'''
Visual-Doom-AI: DoomAI.py
Authors: Rafael Zamora
Last Updated: 12/16/16
CHANGE-LOG:

'''

"""
This class is used to define the Visual Doom AI.

"""

from random import choice
import numpy as np

class DoomAI():

    def __init__(self):
        np.random.seed(123)

    def act(self, buffer_, actions):
        action = choice(actions)
        return action
