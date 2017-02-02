#!/usr/bin/python3
'''
Visual-Doom-AI: DoomSim.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 1/29/17
CHANGE-LOG:
    1/27/17
        - ADDED Comments
    1/29/17
        - BUGFIXED replay .csv saved when save set to false on human_play and ai_play.
    2/2/17
        - BUGFIXED action input size bug

'''

"""
DoomSim is used to run various simulations of the VizDoom engine.
It allows for simulations controlled by both human and AI players
as well as running replay simulations.

Note: Gameplay data is stored in /data/doom_replay_data/ as .lmp and .csv(action history)

"""

import sys, getopt, datetime, itertools
import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution
from DoomAI import DoomAI
from DataProcessor import process_buffer

class DoomSim():

    def __init__(self):
        '''
        Method initializes Vizdoom engine used for simulation.

        Note: Doom level run by the sim is currently hardcoded in the self.doom_map variable

        '''
        self.doom_map = 'map01'
        self.sim = DoomGame()
        self.sim.load_config("configs/doom2_singleplayer.cfg")
        self.sim.set_doom_map(self.doom_map)

    def human_play(self, save=True):
        '''
        Method runs human player Doom simulation at 800 X 600 resolution.
        Gameplay data is saved (if save == True) with filename formatted as:
        player_{doom_map}_{timestamp}.lmp - Vizdoom Replay File
        player_{doom_map}_{timestamp}.csv - Action History (Vizdoom Replay does not store this data natively)

        '''
        date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        filename = "player" + "_" + self.doom_map + "_" + date + ".lmp"

        self.sim.set_mode(Mode.SPECTATOR)
        self.sim.set_screen_resolution(ScreenResolution.RES_800X600)
        self.sim.init()

        actions = []

        if save: self.sim.new_episode("../data/doom_replay_data/" + filename)
        else: self.sim.new_episode
        while not self.sim.is_episode_finished():
            actions.append(self.sim.get_last_action())
            self.sim.advance_action()
        actions.append(self.sim.get_last_action())
        if self.sim.is_player_dead(): actions.append([3 for i in range(self.sim.get_available_buttons_size())])
        else: actions.append([2 for i in range(self.sim.get_available_buttons_size())])
        self.sim.close()

        actions = np.array(actions[1:])
        if save: np.savetxt("../data/doom_replay_data/" + filename[:-3] + "csv", actions, fmt='%i', delimiter=",")

    def ai_play(self, save=True):
        '''
        Method runs AI player Doom simulation at 160 X 120 resolution.
        Gameplay data is saved (if save == True) with filename formatted as:
        ai_{doom_map}_{timestamp}.lmp - Vizdoom Replay File
        ai_{doom_map}_{timestamp}.csv - Action History (Vizdoom Replay does not store this data natively)

        Note: Number of frames simulation runs for is hard code in the cycles variable
              AI behavior is programmed in DoomAI.py

        '''
        date = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        filename = "ai" + "_" + self.doom_map + "_" + date + ".lmp"

        cycles = 1000
        self.sim.set_screen_resolution(ScreenResolution.RES_160X120)
        self.sim.init()

        actions = []
        action_list = self.get_actions(self.sim.get_available_buttons_size())
        ai = DoomAI(action_list)

        if save: self.sim.new_episode("../data/doom_replay_data/" + filename)
        else: self.sim.new_episode
        while not self.sim.is_episode_finished():
            actions.append(self.sim.get_last_action())
            state = self.sim.get_state()
            if state.number == cycles: self.sim.send_game_command('kill')
            ai_action = ai.act(process_buffer(state.screen_buffer, state.depth_buffer))
            self.sim.set_action(list(ai_action))
            self.sim.advance_action()
        actions.append(self.sim.get_last_action())
        if self.sim.is_player_dead(): actions.append([3 for i in range(self.sim.get_available_buttons_size())])
        else: actions.append([2 for i in range(self.sim.get_available_buttons_size())])
        self.sim.close()

        actions = np.array(actions[1:])
        if save: np.savetxt("../data/doom_replay_data/" + filename[:-3] + "csv", actions, fmt='%i', delimiter=",")

    def replay(self, filename):
        '''
        Method runs a replay of the simulations at 800 x 600 simulation.

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
        '''
        Method returns all possible permutaitons of action vectors.

        '''
        actions = list(itertools.product(range(2), repeat=num_of_actions))
        return actions
