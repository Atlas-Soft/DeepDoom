#!/usr/bin/python3
'''
Visual-Doom-AI: Qlearning4k.py
Authors: Rafael Zamora
Last Updated: 2/14/17
CHANGE-LOG:
	2/14/17
		REVISED code from qlearning4k for our project

'''

"""
Citation: Code is based on code from qlearning4k github project

"""
import numpy as np
from random import sample
from keras import backend as K
import matplotlib.pyplot as plt
import os

class QLearn:
	"""
	"""

	def __init__(self, model, memory_size=1000, nb_frames=1):
		'''
		'''
		self.memory = Memory(memory_size)
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	def get_game_data(self, game):
		'''
		Method returns model ready state data

		'''
		frame = game.get_state()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, observe=0, checkpoint=None, filename='w_.h5'):
		'''
		'''
		print("\nQ-Learn Training:\n")
		model = self.model
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else: final_epsilon = epsilon
		loss_history = []
		reward_history = []
		for epoch in range(nb_epoch):
			total_reward = 0
			loss = 0.
			game.reset()
			self.frames = None
			game_over = False
			S = self.get_game_data(game)
			while not game_over:
				if np.random.random() < epsilon or epoch < observe:
					a = int(np.random.randint(game.nb_actions))
				else:
					q = model.predict(S.reshape(1, self.nb_frames, 120, 160))
					a = int(np.argmax(q[0]))
				game.play(a)
				r = game.get_score()
				S_prime = self.get_game_data(game)
				game_over = game.is_over()
				transition = [S, a, r, S_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime
				if epoch >= observe:
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets)[0])
				if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
					model.save_weights('../data/model_weights/' + filename, overwrite=True)
			total_reward = game.get_total_score()
			if epsilon > final_epsilon and epoch >= observe: epsilon -= delta
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Total Reward {}".format(epoch + 1, nb_epoch, loss, epsilon, total_reward))
			reward_history.append(game.get_total_score())
			loss_history.append(loss)

		# summarize history for reward
		plt.plot(reward_history)
		plt.title('Total Reward')
		plt.ylabel('reward')
		plt.xlabel('epoch')
		plt.savefig("../doc/figures/total_reward.png")
		plt.figure()

		# summarize history for loss
		plt.plot(loss_history)
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig("../doc/figures/loss.png")
		plt.show()

class Memory():
    """
    """

    def __init__(self, memory_size=100):
        '''
        '''
        self.memory = []
        self._memory_size = memory_size

    def remember(self, s, a, r, s_prime, game_over):
        '''
        '''
        self.input_shape = s.shape[2:]
        self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), 1 * np.array(game_over).flatten()]))
        if self._memory_size > 0 and len(self.memory) > self._memory_size: self.memory.pop(0)

    def get_batch(self, model, batch_size, gamma=0.9):
        '''
        '''
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        nb_actions = model.output_shape[-1]
        samples = np.array(sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2]
        r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
        game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
        S = S.reshape((batch_size, ) + self.input_shape)
        S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
        X = np.concatenate([S, S_prime], axis=0)
        Y = model.predict(X)
        Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
        delta = np.zeros((batch_size, nb_actions))
        a = np.cast['int'](a)
        delta[np.arange(batch_size), a] = 1
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        return S, targets

class Game(object):

	def __init__(self):
		self.reset()
		self.nb_actions = 0

	def reset(self):
		pass

	def play(self, action):
		pass

	def get_state(self):
		return None

	def get_score(self):
		return 0

	def is_over(self):
		return False

	def get_total_score(self):
		return False
