#!/usr/bin/python3
'''
RLAgent.py
Authors: Rafael Zamora
Last Updated: 3/26/17

'''

"""
This script defines the interface between the Models and the Vizdoom
environment used to train and test Reinforcement Learning Models.

"""
import numpy as np
from Models import *
from random import sample
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import datetime

class RLAgent:
	"""
	RLAgent class interfaces Models with Vizdoom game and preforms training.

	The following are learning algorithms implemented:

	* Deep Q-Learning					(learn_algo = dqlearn)
	* Deep SARSA						(learn_algo = sarsa) ***Not Working
	* Double Deep Q-Learning			(learn_algo = double_dqlearn)


	The following are exploration policies implemented:

	* Epsilon-Greedy	(exp_policy = e-greedy)

	Linear alpha (Reinforcement Learning rate) decay is implemented.

	"""

	def __init__(self, model, learn_algo = 'dqlearn', exp_policy='e-greedy', nb_tests=100, frame_skips=4, nb_epoch=1000, steps=1000, target_update=100,
		batch_size=50, memory_size=1000, nb_frames=1, alpha = [1.0,0.1], alpha_rate=1.0, alpha_wait=0, gamma=0.9, epsilon=[1., .1], epsilon_rate=1.0,
		epislon_wait=0, state_predictor_watch=0):
		'''
		Method initiates learning parameters for Reinforcement Learner.

		'''
		self.model = model
		self.memory = ReplayMemory(memory_size)
		self.prev_frames = None
		self.nb_tests = nb_tests

		# Learning Parameters
		self.learn_algo = learn_algo
		self.exp_policy = exp_policy
		self.nb_epoch = nb_epoch
		self.steps = steps
		self.batch_size = batch_size
		self.nb_frames = nb_frames
		self.frame_skips = frame_skips

		# Set Gamma
		self.gamma = gamma

		# Set Alpha and linear Alpha decay
		self.alpha, self.final_alpha = alpha
		self.alpha_wait = alpha_wait
		self.delta_alpha = ((alpha[0] - alpha[1]) / (nb_epoch * alpha_rate))

		# Set Epsilon and linear Epsilon decay
		self.epsilon, self.final_epsilon = epsilon
		self.epislon_wait = epislon_wait
		self.delta_epsilon =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))

		# Set Double Deep Q-Learning parameters
		if self.learn_algo == "double_dqlearn" or self.learn_algo == "dispersed_double_dqlearn":
			self.model.target_network = Model(input=self.model.x0, output=self.model.y0)
			self.model.target_network.set_weights(self.model.online_network.get_weights())
			self.model.target_network.compile(optimizer=self.model.optimizer, loss=self.model.loss_fun)
			self.target_update = target_update

		# Set Dispersed Double Deep Q-Learning parameters
		if self.learn_algo == "dispersed_double_dqlearn":
			self.state_predictor_watch = state_predictor_watch
			self.state_predictor_loss = 0
			self.state_predictor_batch = None
			if self.model.state_predictor is None:
				self.model.state_predictor = StatePredictionModel(resolution=self.model.resolution, nb_frames=self.model.nb_frames, nb_actions=self.model.nb_actions, depth_radius=self.model.depth_radius, depth_contrast=self.model.depth_contrast)

	def get_state_data(self, game):
		'''
		Method returns model ready state data. The buffers from Vizdoom are
		processed and grouped depending on how many previous frames the model is
		using as defined in the nb_frames variable.

		'''
		frame = game.get_processed_state(self.model.depth_radius, self.model.depth_contrast)
		if self.prev_frames is None:
			self.prev_frames = [frame] * self.nb_frames
		else:
			self.prev_frames.append(frame)
			self.prev_frames.pop(0)
		return np.expand_dims(self.prev_frames, 0)

	def train(self, game):
		'''
		Method preforms Reinforcement Learning on agent's model according to
		learning parameters.

		'''
		print("\nTraining:", game.config_filename)
		print("Model:", self.model.__class__.__name__)
		print("Algorithm:", self.learn_algo)
		print("Exploration_Policy:", self.exp_policy)
		print("Frame Skips:", self.frame_skips)
		print("Number of Previous Frames Used:", self.nb_frames)
		print("Batch Size:", self.batch_size, '\n')

		# Reinforcement Learning Loop
		training_data = []
		best_score = None
		for epoch in range(self.nb_epoch):
			pbar = tqdm(total=self.steps)
			step = 0
			loss = 0
			total_reward = 0
			game.game.new_episode()
			self.prev_frames = None
			if self.model.__class__.__name__ == 'HDQNModel': self.model.sub_model_frames = None
			S = self.get_state_data(game)
			a_prime = 0

			# Preform learning step
			while step < self.steps:

				# Exploration Policies
				if self.exp_policy == 'e-greedy':
					if np.random.random() < self.epsilon:
						q = int(np.random.randint(self.model.nb_actions))
						a = self.model.predict(game, q)
					else:
						q = self.model.online_network.predict(S)
						q = int(np.argmax(q[0]))
						a = self.model.predict(game, q)

				# Advance Action over frame_skips + 1
				if not game.game.is_episode_finished(): game.play(a, self.frame_skips+1)

				r = game.game.get_last_reward()

				if self.model.__class__.__name__ == 'HDQNModel':
					if q >= len(self.model.actions):
						for i in range(self.model.skill_frame_skip):
							a = self.model.predict(game, q)
							if not game.game.is_episode_finished(): game.play(a, self.frame_skips+1)
							r += game.game.get_last_reward()

				# Store transition in memory
				a = q
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
				elif self.learn_algo == 'double_dqlearn':
					batch = self.memory.get_batch_ddqlearn(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)
				elif self.learn_algo == 'dispersed_double_dqlearn':
					batch = self.memory.get_batch_ddqlearn(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)
					self.state_predictor_batch = self.memory.get_batch_state_predictor(model=self.model, batch_size=10)

				# Train model online network
				if batch:
					inputs, targets = batch
					loss += float(self.model.online_network.train_on_batch(inputs, targets))

				# Train State Prediction Model (dispersed DDQ-Learning)
				if self.learn_algo == 'dispersed_double_dqlearn':
					if self.state_predictor_batch:
						inputs, targets = self.state_predictor_batch
						self.state_predictor_loss += float(self.model.state_predictor.autoencoder_network.train_on_batch(inputs, targets))

				# Update Target Network weights (DDQ-Learning)
				if self.model.target_network and step % self.target_update == 0:
					self.model.target_network.set_weights(self.model.online_network.get_weights())

				if game_over:
					if self.model.__class__.__name__ == 'HDQNModel': self.model.sub_model_frames = None
					game.game.new_episode()
					self.prev_frames = None
					S = self.get_state_data(game)
				step += 1
				pbar.update(1)

			# Decay Epsilon
			if self.epsilon > self.final_epsilon and epoch >= self.epislon_wait: self.epsilon -= self.delta_epsilon

			# Decay Alpha
			if self.alpha > self.final_alpha and epoch >= self.alpha_wait: self.alpha -= self.delta_alpha

			# Run Tests
			print("Testing:")
			pbar.close()
			pbar = tqdm(total=self.nb_tests)
			rewards = []
			for i in range(self.nb_tests):
				rewards.append(game.run(self))
				pbar.update(1)
			rewards = np.array(rewards)
			training_data.append([loss, np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards)])
			np.savetxt("../data/results/"+ self.learn_algo.replace("_","-")+'_'+ self.model.__class__.__name__+'_'+ game.config_filename[:-4].replace("_","-") + ".csv", np.array(training_data))

			# Save best weights
			total_reward_avg = training_data[-1][1]
			if best_score is None or (best_score is not None and total_reward_avg > best_score):
				self.model.save_weights(self.learn_algo+'_'+ self.model.__class__.__name__+'_'+ game.config_filename[:-4] + ".h5")
				best_score = total_reward_avg

			if self.model.state_predictor: self.model.state_predictor.save_weights('sp_' + self.learn_algo+'_'+ self.model.__class__.__name__+'_'+ game.config_filename[:-4] + ".h5")

			# Print Epoch Summary
			if self.learn_algo == 'dispersed_double_dqlearn':
				print("Epoch {:03d}/{:03d} | Loss {:.4f} | SP-Loss {:.4f} | Alpha {:.3f} | Epsilon {:.3f} | Average Reward {}".format(epoch + 1, self.nb_epoch, loss, self.state_predictor_loss, self.alpha, self.epsilon, total_reward_avg))
			else:
				print("Epoch {:03d}/{:03d} | Loss {:.4f} | Alpha {:.3f} | Epsilon {:.3f} | Average Reward {}".format(epoch + 1, self.nb_epoch, loss, self.alpha, self.epsilon, total_reward_avg))

		print("Training Finished.\nBest Average Reward:", best_score)

	def transfer_train(self, student_agent, game):
		'''
		Method preforms transfer learning from agent model to desired student model.

		'''
		print("\nTransfer Training:", game.config_filename)
		print("Teacher Model:", self.model.__class__.__name__)
		print("Student Model:", student_agent.model.__class__.__name__)
		print("Frame Skips:", self.frame_skips)
		print("Number of Previous Frames Used", self.nb_frames)
		print("Batch Size:", self.batch_size)

		# Transfer Learning Loop
		training_data = []
		best_score = None
		for epoch in range(self.nb_epoch):
			pbar = tqdm(total=self.steps)
			step = 0
			loss = 0.
			total_reward = 0
			game.game.new_episode()
			self.prev_frames = None
			if self.model.__class__.__name__ == 'HDQNModel': self.model.sub_model_frames = None
			S = self.get_state_data(game)
			a_prime = 0

			# Preform learning step
			while step < self.steps:
				if self.model.__class__.__name__ == 'HDQNModel': self.model.update_submodel_frames(game)

				# Exploration Policies
				if self.exp_policy == 'e-greedy':
					if np.random.random() < self.epsilon:
						q = int(np.random.randint(self.model.nb_actions))
					else:
						q = None

				# Gather Data
				targets = []
				inputs = []
				t, q = self.model.softmax_q_values(S, student_agent.model.actions, q_=q)
				targets.append(t)
				inputs.append(S[0])
				a = student_agent.model.actions[int(np.argmax(t))]

				# Advance Action over frame_skips + 1
				if not game.game.is_episode_finished(): game.play(a, self.frame_skips+1)

				if self.model.__class__.__name__ == 'HDQNModel':
					if q >= len(self.model.actions):
						for i in range(self.model.skill_frame_skip):
							self.model.update_submodel_frames(game)
							S = self.get_state_data(game)
							t, q = self.model.softmax_q_values(S, student_agent.model.actions, q_=q)
							targets.append(t)
							inputs.append(S[0])
							a = student_agent.model.actions[int(np.argmax(t))]
							if not game.game.is_episode_finished(): game.play(a, self.frame_skips+1)

				inputs = np.array(inputs)
				targets = np.array(targets)
				loss += float(student_agent.model.online_network.train_on_batch(inputs, targets))

				S = self.get_state_data(game)
				if game.game.is_episode_finished():
					break
					if self.model.__class__.__name__ == 'HDQNModel': self.model.sub_model_frames = None
					game.game.new_episode()
					self.prev_frames = None
					S = self.get_state_data(game)
				step += 1
				pbar.update(1)

			# Decay Epsilon
			if self.final_epsilon < self.epsilon and epoch >= self.epislon_wait: self.epsilon -= self.delta_epsilon

			# Preform test for epoch
			print("Testing:")
			pbar.close()
			pbar = tqdm(total=self.nb_tests)
			rewards = []
			for i in range(self.nb_tests):
				rewards.append(game.run(student_agent))
				pbar.update(1)
			rewards = np.array(rewards)
			training_data.append([loss, np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards)])
			np.savetxt("../data/results/"+'distilled_'+ self.model.__class__.__name__+'_'+ game.config_filename[:-4].replace("_","-") + ".csv", np.array(training_data))

			# Save best weights
			total_reward_avg = training_data[-1][1]
			if best_score is None or (best_score is not None and total_reward_avg > best_score):
				student_agent.model.save_weights('distilled_'+ self.model.__class__.__name__+'_'+ game.config_filename[:-4] + ".h5")
				best_score = total_reward_avg

			# Print Epoch Summary
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.3f} | Average Reward {}".format(epoch + 1, self.nb_epoch, loss, self.epsilon, total_reward_avg))

		print("Training Finished.\nBest Average Reward:", best_score)

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
		Method generates batch for Deep Double Q-learn training.

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
		a_prime = samples[:, 2 * input_dim + 2]
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

	def get_batch_ddqlearn(self, model, batch_size, alpha=0.01, gamma=0.9):
		'''
		Method generates batch for Double Deep Q-learn training.

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
		a_prime = samples[:, 2 * input_dim + 2]
		game_over = samples[:, 2 * input_dim + 3]
		r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
		game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
		S = S.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)

		# Predict Q-Values
		X = np.concatenate([S, S_prime], axis=0)
		Y = model.online_network.predict(X)
		best = np.argmax(Y[batch_size:], axis = 1)
		YY = model.target_network.predict(S_prime)

		# Get max Q-value
		Qsa = YY.flatten()[np.arange(batch_size)*nb_actions + best].repeat(nb_actions).reshape((batch_size, nb_actions))
		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)
		delta[np.arange(batch_size), a] = 1

		# Get target Q-Values
		targets = ((1 - delta) * Y[:batch_size]) + ((alpha * ((delta * (r + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size]))) + (delta * Y[:batch_size]))
		return S, targets

	def get_batch_state_predictor(self, model, batch_size):
		'''
		Method generates batch for Dispersed Double Deep Q-learn training.

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
		S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
		S = S.reshape((batch_size, ) + self.input_shape)

		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)
		delta[np.arange(batch_size), a] = 1
		a = delta

		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime[np.arange(batch_size), -1]
		S_prime = S_prime.reshape(S_prime.shape[0], 1, S_prime.shape[1], S_prime.shape[2])

		inputs = [S, a]

		return inputs, S_prime
