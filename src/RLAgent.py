#!/usr/bin/python3
'''
RLAgent.py
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
import datetime

class RLAgent:
	"""
	RLAgent class used to interface models with Vizdoom game and
	preform training.

	"""

	def __init__(self, model, learn_algo = 'qlearn', exp_policy='e-greedy', frame_skips=4, nb_epoch=1000, steps=1000, batch_size=50, memory_size=1000, nb_frames=1, alpha = [1.0,0.1], alpha_rate=1.0, alpha_wait=0, gamma=0.9, epsilon=[1., .1], epsilon_rate=1.0, epislon_wait=0, checkpoint=None, filename='w_.h5'):
		'''
		Method initiates learning parameters for Reinforcement Learner.

		'''
		self.model = model
		self.memory = ReplayMemory(memory_size)
		self.frames = None
		self.checkpoint = checkpoint
		self.filename = filename

		# Learning Parameters
		self.learn_algo = learn_algo
		self.exp_policy = exp_policy
		self.nb_epoch = nb_epoch
		self.steps = steps
		self.batch_size = batch_size
		self.nb_frames = nb_frames
		self.frame_skips = frame_skips
		self.gamma = gamma

		# Set Alpha and Alpha decay
		self.alpha, self.final_alpha = alpha
		self.alpha_wait = alpha_wait
		self.delta_alpha = ((alpha[0] - alpha[1]) / (nb_epoch * alpha_rate))

		# Set Epsilon and Epsilon decay
		self.epsilon, self.final_epsilon = epsilon
		self.epislon_wait = epislon_wait
		self.delta_epsilon =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))

	def get_state_data(self, game):
		'''
		Method returns model ready state data.

		'''
		frame = game.get_processed_state(self.model.depth_radius, self.model.depth_contrast)
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def train(self, game):
		'''
		Method preforms Reinforcement Learning on agent's model according to
		learning parameters.

		'''
		loss_history = []
		reward_history = []

		# Q-Learning Loop
		print("\nTraining:", game.config)
		print("Algorithm:", self.learn_algo)
		print("Exploration_Policy:", self.exp_policy, '\n')
		for epoch in range(self.nb_epoch):
			pbar = tqdm(total=self.steps)
			step = 0
			loss = 0.
			total_reward = 0
			game.game.new_episode()
			self.frames = None
			S = self.get_state_data(game)
			a_prime = 0

			# Preform learning step
			while step < self.steps:

				# Exploration Policies
				if self.exp_policy == 'e-greedy':
					if np.random.random() < self.epsilon or epoch < self.epislon_wait:
						q = int(np.random.randint(len(game.actions)))
						a = self.model.predict(game, q)
					else:
						q = self.model.online_network.predict(S)
						q = int(np.argmax(q[0]))
						a = self.model.predict(game, q)

				# Advance Action over frame_skips + 1
				for i in range(self.frame_skips+1):
					if not game.game.is_episode_finished(): game.play(a)

				# Store transition in memory
				r = game.game.get_last_reward()
				S_prime = self.get_state_data(game)
				game_over = game.game.is_episode_finished()
				transition = [S, a, r, S_prime, a_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime
				a_prime = a

				# Generate training batch
				if self.learn_algo == 'dqlearn':
					batch = self.memory.get_batch_dqlearn(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)
				elif self.learn_algo == 'sarsa':
					batch = self.memory.get_batch_sarsa(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)

				# Train model online network
				if batch:
					inputs, targets = batch
					loss += float(self.model.online_network.train_on_batch(inputs, targets))

				if game_over:
					game.game.new_episode()
					self.frames = None
					S = self.get_state_data(game)
				step += 1
				pbar.update(1)

			# Save weights at checkpoints
			if self.checkpoint and ((epoch + 1 - self.epislon_wait) % self.checkpoint == 0 or epoch + 1 == self.nb_epoch):
				self.model.save_weights(self.filename)

			# Decay Epsilon
			if self.epsilon > self.final_epsilon and epoch >= self.epislon_wait: self.epsilon -= self.delta_epsilon

			# Decay Alpha
			if self.alpha > self.final_alpha and epoch >= self.alpha_wait: self.alpha -= self.delta_alpha

			# Preform test for epoch
			print("Testing:")
			pbar.close()
			pbar = tqdm(total=100)
			for i in range(100):
				total_reward += game.run(self)
				pbar.update(1)
			total_reward /= 100
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Alpha {:.3f} | Epsilon {:.3f} | Average Reward {}".format(epoch + 1, self.nb_epoch, loss, self.alpha, self.epsilon, total_reward))
			reward_history.append(total_reward)
			loss_history.append(loss)

		# Summarize history for reward
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

class ReplayMemory():
	"""
	ReplayMemory class used to stores transition data and generate batces for Q-learning.

	"""
	def __init__(self, memory_size=100):
		'''
		Method initiates memory class.

		'''
		self.memory = []
		self._memory_size = memory_size

	def remember(self, s, a, r, s_prime, a_prime, game_over):
		'''
		Method stores flattened stransition to memory bank.

		'''
		self.input_shape = s.shape[1:]
		self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), np.array(a_prime).flatten(), 1 * np.array(game_over).flatten()]))
		if self._memory_size > 0 and len(self.memory) > self._memory_size: self.memory.pop(0)

	def get_batch_dqlearn(self, model, batch_size, alpha=1.0, gamma=0.9):
		'''
		Method generates batch for Deep Q-learn training.
		'''
		nb_actions = model.online_network.output_shape[-1]
		input_dim = np.prod(self.input_shape)

		# Generate Sample
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		samples = np.array(sample(self.memory, batch_size))

		# Restructure Data
		S = samples[:, 0 : input_dim]
		a = samples[:, input_dim]
		r = samples[:, input_dim + 1]
		S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
		game_over = samples[:, 2 * input_dim + 3]
		r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
		game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
		S = S.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)

		# Predict Q-Values
		X = np.concatenate([S, S_prime], axis=0)
		Y = model.online_network.predict(X)

		# Get max Q-value
		Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)
		delta[np.arange(batch_size), a] = 1

		# Get target Q-Values
		targets = ((1 - delta) * Y[:batch_size]) + ((alpha * ((delta * (r + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size]))) + (delta * Y[:batch_size]))
		return S, targets

	def get_batch_sarsa(self, model, batch_size, alpha=1.0, gamma=0.9):
		'''
		Method generates batch for Deep SARSA training.
		'''
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		nb_actions = model.online_network.output_shape[-1]
		samples = np.array(sample(self.memory, batch_size))
		input_dim = np.prod(self.input_shape)
		S = samples[:, 0 : input_dim]
		a = samples[:, input_dim]
		r = samples[:, input_dim + 1]
		S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
		a_prime = samples[:, 2 * input_dim + 2]
		game_over = samples[:, 2 * input_dim + 3]
		r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
		game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
		S = S.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
		X = np.concatenate([S, S_prime], axis=0)
		Y = model.online_network.predict(X)
		Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)
		delta[np.arange(batch_size), a] = 1
		targets = ((1 - delta) * Y[:batch_size]) + ((alpha * ((delta * (r + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size]))) + (delta * Y[:batch_size]))
		return S, targets
