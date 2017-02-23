#!/usr/bin/python3
'''
Doom.py
Authors: Rafael Zamora
Last Updated: 2/18/17

'''

from Qlearning4k import Game
from vizdoom import DoomGame, Mode, ScreenResolution
import itertools as it
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from tqdm import tqdm

class Doom(Game):

    def __init__(self, config, frame_skips=0, depth_radius=0.225, depth_contrast=0.9):
        self.config = config
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(False)
        self.game.init()

        self.res = (self.game.get_screen_height(), self.game.get_screen_width())
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast

        button_count = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=button_count)]
        self.frame_tics = 1 + frame_skips
        self.pbar = None
        self.game.new_episode()

    def reset(self):
        self.game.new_episode()

    def play(self, action):
        self.game.set_action(self.actions[action])
        for i in range(self.frame_tics):
            if not self.is_over():
                self.game.advance_action()
        if self.pbar: self.pbar.update(self.frame_tics)

    def get_state(self):
        state = self.game.get_state()
        screen_buffer = np.array(state.screen_buffer).astype('float32')/255
        try:
            grey_buffer = np.dot(np.transpose(screen_buffer, (1, 2, 0)), [0.21, 0.72, 0.07])#Greyscaling
            depth_buffer = np.array(state.depth_buffer).astype('float32')/255
            depth_buffer[(depth_buffer > self.depth_radius)] = self.depth_radius #Effects depth radius
            depth_buffer_filtered = (depth_buffer - np.amin(depth_buffer))/ (np.amax(depth_buffer) - np.amin(depth_buffer))
            processed_buffer = ((1 - self.depth_contrast) * grey_buffer) + (self.depth_contrast* (1- depth_buffer))
            processed_buffer = (processed_buffer - np.amin(processed_buffer))/ (np.amax(processed_buffer) - np.amin(processed_buffer))
            processed_buffer = np.round(processed_buffer, 6)
            processed_buffer = processed_buffer.reshape(self.res[-2:])
        except:
            processed_buffer = np.zeros(self.res[-2:])

        return processed_buffer

    def get_score(self):
	    return self.game.get_last_reward()

    def is_over(self):
        return self.game.is_episode_finished()

    def get_total_score(self):
        return self.game.get_total_reward()

    def run(self, agent, save_replay='', verbose=False):
        '''
        Method runs a instance of Doom.

        '''
        self.game.close()
        self.game.set_window_visible(False)
        self.game.add_game_args("+vid_forcesurface 1")
        self.game.init()
        if verbose: print("\nRunning Simulation:", self.config)

        if save_replay != '': self.game.new_episode("../data/replay_data/" + save_replay)
        else: self.game.new_episode()
        if verbose: self.pbar = tqdm(total=self.game.get_episode_timeout())
        while not self.is_over():
            S = agent.get_game_data(self)
            q = agent.model.online_network.predict(S)
            q = int(np.argmax(q[0]))
            a = agent.model.predict(S, q)
            self.play(a)

        score = self.game.get_total_reward()
        if verbose:
            print("Total Score:", score)
            self.pbar.close()
        return score

    def replay(self, filename, verbose=False):
        '''
        Method runs a replay of the simulations at 800 x 600 simulation.

        '''
        self.game.close()
        self.game.set_screen_resolution(ScreenResolution.RES_800X500)
        self.game.set_window_visible(True)
        self.game.set_ticrate(60)
        self.game.add_game_args("+vid_forcesurface 1")

        self.game.init()
        print("\nRunning Replay:", filename)
        self.game.replay_episode("../data/replay_data/" + filename)
        while not self.game.is_episode_finished():
            if verbose: print("Reward:", self.game.get_last_reward())
            self.game.advance_action()

        score = self.game.get_total_reward()
        print("Total Score:", score)
        self.game.close()

    def human_play(self):
        '''
        '''
        self.game.close()
        self.game.set_mode(Mode.SPECTATOR)
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.set_ticrate(30)
        self.game.init()

        self.game.new_episode()
        while not self.game.is_episode_finished():
            print(self.game.get_total_reward())
            self.game.advance_action()
        self.game.close()
