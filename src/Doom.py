from Qlearning4k import Game
from vizdoom import DoomGame, Mode, ScreenResolution
import itertools as it
import numpy as np
from tqdm import tqdm

class Doom(Game):

    def __init__(self, config, frame_tics = 1):
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)
        self.game.init()

        button_count = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=button_count)]
        self.nb_actions = len(self.actions)
        self.frame_tics = frame_tics
        self.game.new_episode()

    def reset(self):
        self.game.new_episode()
        self.pbar = tqdm(total=self.game.get_episode_timeout())

    def play(self, action):
        self.game.set_action(self.actions[action])
        for i in range(self.frame_tics):
            if not self.is_over():
                self.game.advance_action()
                self.pbar.update(1)

    def get_state(self):
        state = self.game.get_state()
        try:
            screen_buffer = np.array(state.screen_buffer).astype('float32')/255
            depth_buffer = np.array(state.depth_buffer).astype('float32')/255
            grey_buffer = np.dot(np.transpose(screen_buffer, (1, 2, 0)), [0.21, 0.72, 0.07])#Greyscaling
            depth_buffer[(depth_buffer > .25)] = .25 #Effects depth radius
            depth_buffer_filtered = (depth_buffer - np.amin(depth_buffer))/ (np.amax(depth_buffer) - np.amin(depth_buffer))
            processed_buffer = grey_buffer + (.75* (1- depth_buffer_filtered))
            processed_buffer = (processed_buffer - np.amin(processed_buffer))/ (np.amax(processed_buffer) - np.amin(processed_buffer))
            processed_buffer = np.round(processed_buffer, 6)
            processed_buffer = processed_buffer.reshape(1, 120, 160)
        except:
            processed_buffer = np.zeros((1, 120, 160))
        return processed_buffer

    def get_score(self):
	    return self.game.get_last_reward()

    def is_over(self):
        return self.game.is_episode_finished()

    def get_total_score(self):
        return self.game.get_total_reward()

    def run(self, agent, save_replay=''):
        '''
        '''
        self.game.close()
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(True)
        self.game.add_game_args("+vid_forcesurface 1")
        self.game.init()
        print("Running Simulation:")

        if save_replay == '': self.game.new_episode("../data/replay_data/" + filename)
        else: self.game.new_episode()
        while not self.is_over():
            q = model.predict(S.reshape(1, 1, 120, 160))
            a = int(np.argmax(q[0]))
            self.play(a)

        score = self.game.get_total_reward()
        print("Total Score:", score)
        self.game.close()

    def replay(self, filename):
        '''
        Method runs a replay of the simulations at 800 x 600 simulation.

        '''
        self.game.close()
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.add_game_args("+vid_forcesurface 1")

        self.game.init()
        print("Running Replay:")
        self.game.replay_episode("../data/replay_data/" + filename)
        while not self.game.is_episode_finished():
            self.game.advance_action()

        score = self.game.get_total_reward()
        print("Total Score:", score)
        self.game.close()
