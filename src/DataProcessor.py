#!/usr/bin/python3
'''
Visual-Doom-AI: DataProcessor.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 1/27/17
CHANGE-LOG:
    1/27/17
        - ADDED class/method comments with Rafael Zamora.
    1/29/17
        - EDITED comments

'''

"""
DataProcessor processes and loads buffer, action, and reward data for running replays. Data files are encoded in a json format
in the '/data/doom_processed_data'. Resulting black screens signify death, while resulting white screens signify a level complete.

"""
import os, sys, getopt, datetime, json, base64
import numpy as np
from vizdoom import DoomGame, ScreenResolution, GameVariable

class DataProcessor():

    def __init__(self):
        '''
        Method initializes the game configuration settings, such as enemy skill level and available buttons
        from 'configs/doom2_singleplayer.cfg', as well as the smaller screen resolution for replays.

        '''
        self.sim = DoomGame()
        self.sim.load_config("configs/doom2_singleplayer.cfg")
        self.sim.set_screen_resolution(ScreenResolution.RES_160X120)
        self.sim.set_window_visible(False)
        self.sim.set_available_game_variables([GameVariable.POSITION_X, GameVariable.POSITION_Y])

    def process_replays(self):
        '''
        Method processes the specified replay file from '/data/doom_replay_data/'
        and produces a file stored in '/data/doom_processed_data' for training.

        '''
        for filename in os.listdir("../data/doom_replay_data/"):
            if filename.endswith(".lmp"):
                print("Processing:", filename)
                screen_buffers = []
                depth_buffers = []
                actions = np.genfromtxt("../data/doom_replay_data/" + filename[:-3] + "csv", delimiter=',').astype(int)
                rewards = []

                #Gather data from replay
                self.sim.init()
                self.sim.replay_episode("../data/doom_replay_data/" + filename)
                while not self.sim.is_episode_finished():
                    state = self.sim.get_state()
                    if state.number < len(actions):
                        screen_buffers.append(state.screen_buffer)
                        depth_buffers.append(state.depth_buffer)
                        rewards.append(self.level1_reward(state.game_variables))
                    else: break
                    self.sim.advance_action()
                self.sim.close()

                screen_buffers = np.array(screen_buffers)
                depth_buffers = np.array(depth_buffers)

                #Process buffers
                processed_buffers = []
                for i in range(len(screen_buffers)):
                    processed_buffers.append(process_buffer(screen_buffers[i], depth_buffers[i]))

                #Add Final State Marker
                if actions[-1][0] == 2:
                    processed_buffers.append(np.ones((1, 120, 160)))
                    rewards.append(1)
                elif actions[-1][0] == 3:
                    processed_buffers.append(np.zeros((1, 120, 160)))
                    rewards.append(0)
                processed_buffers = np.array(processed_buffers)
                rewards = np.array(rewards)

                #Encode and write to file
                with open("../data/doom_processed_data/" + filename[:-3] + "json", "w") as f:
                    for i in range(len(processed_buffers)):
                        f.write(b64encode_data(processed_buffers[i], actions[i], rewards[i]) + "\n")
                print("Done.")

    def level1_reward(self, game_vars):
        return 0

def process_buffer(screen_buffer, depth_buffer):
    '''
    Method recieves three channels from the screen buffer (rgb) and one channel from the depth buffer. Normalizing the screen buffers
    and applying the depth buffer onto it creates a single, filtered, gray-scaled channel that gets returned for training purposes.

    '''
    depth_buffer_float = depth_buffer.astype('float32')/255
    screen_buffer_float = screen_buffer.astype('float32')/255
    grey_buffer = np.dot(np.transpose(screen_buffer_float, (1, 2, 0)), [0.21, 0.72, 0.07])#Greyscaling
    depth_buffer_float[(depth_buffer_float > .25)] = .25 #Effects depth radius
    depth_buffer_filtered = (depth_buffer_float - np.amin(depth_buffer_float))/ (np.amax(depth_buffer_float) - np.amin(depth_buffer_float))
    processed_buffer = grey_buffer + (.75* (1- depth_buffer_filtered))
    processed_buffer = (processed_buffer - np.amin(processed_buffer))/ (np.amax(processed_buffer) - np.amin(processed_buffer))
    processed_buffer = np.round(processed_buffer, 6)
    processed_buffer = processed_buffer.reshape(1, 120, 160)
    return processed_buffer

def b64encode_data(buffer_, action, reward):
    '''
    Method encodes buffer, action and reward data using base64 encoding.
    Encoded data is stored as JSON.

    '''
    data = {}
    data['buffer'] = [str(buffer_.dtype), base64.b64encode(buffer_).decode('utf-8'), buffer_.shape]
    data['action'] = [str(action.dtype), base64.b64encode(action).decode('utf-8'), action.shape]
    data['reward'] = [str(reward.dtype), base64.b64encode(reward).decode('utf-8'), reward.shape]
    return json.dumps(data)

def b64decode_data(json_dump):
    '''
    Method decodes buffer, action and reward data from JSON.
    Uses base64 decoding.

    '''
    data = json.loads(json_dump)
    buffer_ = np.frombuffer(base64.b64decode(data['buffer'][1]),np.dtype(data['buffer'][0])).reshape(data['buffer'][2])
    action = np.frombuffer(base64.b64decode(data['action'][1]),np.dtype(data['action'][0])).reshape(data['action'][2])
    reward = np.frombuffer(base64.b64decode(data['reward'][1]),np.dtype(data['reward'][0])).reshape(data['reward'][2])
    return buffer_, action, reward

def load_data(filename):
    '''
    Method loads processed data stored in designated file.
    Returns buffer, action and reward data.
    
    '''
    buffers = []
    actions = []
    rewards = []
    with open("../data/doom_processed_data/" + filename, "r") as f:
        for json_dump in f:
            buffer_, action, reward = b64decode_data(json_dump)
            buffers.append(buffer_)
            actions.append(action)
            rewards.append(reward)
    buffers = np.array(buffers)
    actions = np.array(actions)
    rewards = np.array(rewards)
    return buffers, actions, rewards
