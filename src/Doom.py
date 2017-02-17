from Qlearning4k import Game
from vizdoom import DoomGame, Mode, ScreenResolution
import itertools as it
import numpy as np
from tqdm import tqdm

class Doom(Game):

    def __init__(self, config, frame_tics = 1):
        self.config = config
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)
        self.game.init()

        button_count = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=button_count)]
        self.frame_tics = frame_tics
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
        try:
            screen_buffer = np.array(state.screen_buffer).astype('float32')/255
            depth_buffer = np.array(state.depth_buffer).astype('float32')/255
            processed_buffer = np.concatenate([screen_buffer, depth_buffer], 0)
        except:
            processed_buffer = np.zeros((4, 120, 160))
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
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)
        self.game.add_game_args("+vid_forcesurface 1")
        self.game.init()
        if verbose: print("\nRunning Simulation:", self.config)

        if save_replay != '': self.game.new_episode("../data/replay_data/" + save_replay)
        else: self.game.new_episode()
        if verbose: self.pbar = tqdm(total=self.game.get_episode_timeout())
        while not self.is_over():
            S = agent.get_game_data(self)
            q = agent.model.model.predict(S.reshape(1, agent.nb_frames, 120, 160))
            q = int(np.argmax(q[0]))
            a = agent.model.predict(S.reshape(1, agent.nb_frames, 120, 160), q)
            self.play(a)

        score = self.game.get_total_reward()
        if verbose:
            print("Total Score:", score)
            self.pbar.close()
        return score

    def replay(self, filename):
        '''
        Method runs a replay of the simulations at 800 x 600 simulation.

        '''
        self.game.close()
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.set_ticrate(60)
        self.game.add_game_args("+vid_forcesurface 1")

        self.game.init()
        print("\nRunning Replay:", filename)
        self.game.replay_episode("../data/replay_data/" + filename)
        while not self.game.is_episode_finished():
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
