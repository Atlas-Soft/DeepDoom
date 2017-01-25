#!/usr/bin/python3
'''
Visual-Doom-AI:
Authors:
Last Updated:
CHANGE-LOG:

'''

"""

"""

import sys, getopt, datetime, itertools
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution
from DoomAI import DoomAI
from DataProcessor import process_buffer

class DoomSim():

    def __init__(self):
        self.doom_map = 'map01'
        self.sim = DoomGame()
        self.sim.load_config("configs/doom2_singleplayer.cfg")
        self.sim.set_doom_map(self.doom_map)

    def human_play(self):
        '''

        '''
        date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        filename = "player" + "_" + self.doom_map + "_" + date + ".lmp"

        self.sim.set_mode(Mode.SPECTATOR)
        self.sim.set_screen_resolution(ScreenResolution.RES_800X600)
        self.sim.init()

        actions = []

        self.sim.new_episode("../data/doom_replay_data/" + filename)
        while not self.sim.is_episode_finished():
            actions.append(self.sim.get_last_action())
            self.sim.advance_action()
        actions.append(self.sim.get_last_action())
        self.sim.close()

        actions = np.array(actions)
        np.savetxt("../data/doom_replay_data/" + filename[:-3] + "csv", actions, fmt='%i', delimiter=",")

    def ai_play(self):
        '''

        '''
        date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        filename = "ai" + "_" + self.doom_map + "_" + date + ".lmp"

        self.sim.set_episode_timeout(50)
        self.sim.set_screen_resolution(ScreenResolution.RES_160X120)
        self.sim.init()

        ai = DoomAI()
        actions = []
        action_list = self.get_actions(self.sim.get_available_buttons_size())

        self.sim.new_episode("../data/doom_replay_data/" + filename)
        while not self.sim.is_episode_finished():
            actions.append(self.sim.get_last_action())
            state = self.sim.get_state()
            ai_action = ai.act(process_buffer(state.screen_buffer, state.depth_buffer), action_list)
            self.sim.set_action(list(ai_action))
            self.sim.advance_action()
        actions.append(self.sim.get_last_action())
        self.sim.close()

        actions = np.array(actions)
        np.savetxt("../data/doom_replay_data/" + filename[:-3] + "csv", actions, fmt='%i', delimiter=",")

    def replay(self, filename):
        '''

        '''
        self.sim.set_screen_resolution(ScreenResolution.RES_800X600)
        self.sim.init()

        actions = np.genfromtxt("../data/doom_replay_data/" + filename[:-3] + "csv", delimiter=',').astype(int)

        self.sim.init()
        self.sim.replay_episode("../data/doom_replay_data/" + filename)
        while not self.sim.is_episode_finished():
            state = self.sim.get_state()
            if state.number > len(actions): break
            self.sim.advance_action()
        self.sim.close()

    def get_actions(self, num_of_actions):
        actions = list(itertools.product(range(2), repeat=num_of_actions))
        return actions
