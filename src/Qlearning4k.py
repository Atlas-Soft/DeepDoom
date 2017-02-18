#!/usr/bin/python3
'''
Qlearning4k.py
Authors: Rafael Zamora
Last Updated: 2/18/17

'''

"""
Citation: Code is based on code from qlearning4k github project

"""
import numpy as np
from random import sample
from keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class QLearnAgent:
	"""
	Q-learner agent class used to interface models with Vizdoom game and
	preform training.

	"""

	def __init__(self, model, nb_epoch=1000, steps=1000, batch_size=50, memory_size=1000, nb_frames=1, alpha = 0.01, gamma=0.9, epsilon=[1., .1], epsilon_rate=1.0, observe=0, checkpoint=None, filename='w_.h5'):
		'''
		Method initiates memory bank for q-learner.

		'''
		self.model = model
		self.nb_epoch = nb_epoch
		self.frames = None
		self.memory = Memory(memory_size)
		self.nb_frames = nb_frames
		self.steps = steps
		self.batch_size = batch_size
		self.nb_frames = nb_frames
		self.alpha = alpha
		self.gamma = gamma

		# Set Epsilon and Epsilon decay
		self.delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
		self.epsilon, self.final_epsilon = epsilon

		self.observe = observe
		self.checkpoint = checkpoint
		self.filename = filename

	def get_game_data(self, game):
		'''
		Method returns model ready state data.

		'''
		frame = game.get_state()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def train(self, game):
		'''
		Method preforms Q learning on agent.

		'''
		loss_history = []
		reward_history = []

		# Q-Learning Loop
		print("\nQ-Learn Training:", game.config, "\n")
		for epoch in range(self.nb_epoch):
			pbar = tqdm(total=self.steps)
			step = 0
			loss = 0.
			total_reward = 0
			game.reset()
			self.frames = None
			S = self.get_game_data(game)

			# Preform learning step
			while step < self.steps:
				if np.random.random() < self.epsilon or epoch < self.observe:
					q = int(np.random.randint(len(game.actions)))
					a = self.model.predict(S, q)
				else:
					q = self.model.q_net.predict(S)
					q = int(np.argmax(q[0]))
					a = self.model.predict(S, q)
				game.play(a)
				r = game.get_score()
				S_prime = self.get_game_data(game)
				game_over = game.is_over()
				transition = [S, a, r, S_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime

				# Generate training batch and train model
				batch = self.memory.get_batch(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)
				if batch:
					inputs, targets = batch
					loss += float(self.model.q_net.train_on_batch(inputs, targets))
				if game_over:
					game.reset()
					self.frames = None
					S = self.get_game_data(game)
				step += 1
				pbar.update(1)

			# Save weights at checkpoints
			if self.checkpoint and ((epoch + 1 - self.observe) % self.checkpoint == 0 or epoch + 1 == self.nb_epoch):
				self.model.save_weights(self.filename)

			# Decay Epsilon
			if self.epsilon > self.final_epsilon and epoch >= self.observe: self.epsilon -= self.delta

			# Preform test for Epoch
			print("Testing:")
			pbar.close()
			pbar = tqdm(total=100)
			for i in range(100):
				total_reward += game.run(self)
				pbar.update(1)
			total_reward /= 100
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Average Reward {}".format(epoch + 1, self.nb_epoch, loss, self.epsilon, total_reward))
			reward_history.append(total_reward)
			loss_history.append(loss)

		# summarize history for reward
		plt.plot(reward_history)
		plt.title('Total Reward')
		plt.ylabel('reward')
		plt.xlabel('epoch')
		plt.savefig("../doc/figures/" + self.filename[:-3] + "_total_reward.png")
		plt.figure()

		# summarize history for loss
		plt.plot(loss_history)
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.savefig("../doc/figures/" + self.filename[:-3] + "_loss.png")
		plt.show()

class Memory():
	"""
	Memory class used to stores transition data and generate batces for Q-learning.

	"""
	def __init__(self, memory_size=100):
		'''
		Method initiates memory class.

		'''
		self.memory = []
		self._memory_size = memory_size

	def remember(self, s, a, r, s_prime, game_over):
		'''
		Method stores transition to memory bank.

		'''
		self.input_shape = s.shape[1:]
		self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), 1 * np.array(game_over).flatten()]))
		if self._memory_size > 0 and len(self.memory) > self._memory_size:
			self.memory.pop(0)

	def get_batch(self, model, batch_size, alpha=0.01, gamma=0.9):
		'''
		Method generates batch for Deep Q-learn training.
		'''
		model = model.q_net
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
		targets = ((1 - delta) * Y[:batch_size]) + ((alpha * ((delta * (r + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size]))) + (delta * Y[:batch_size]))
		return S, targets

class Game(object):

	def __init__(self):
		self.reset()

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
