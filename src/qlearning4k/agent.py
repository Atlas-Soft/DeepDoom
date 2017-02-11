from .memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

class Agent:

	def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
		assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)
		if not nb_frames and not model.input_shape[1]:
			raise Exception("Missing argument : nb_frames not provided")
		elif not nb_frames:
			nb_frames = model.input_shape[1]
		elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
			raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.exp_replay.reset_memory()

	def check_game_compatibility(self, game):
		game_output_shape = (1, None) + game.get_frame().shape
		if len(game_output_shape) != len(self.model.input_shape):
			raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		else:
			for i in range(len(self.model.input_shape)):
				if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != game_output_shape[i]:
					raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		if len(self.model.output_shape) != 2 or self.model.output_shape[1] != game.nb_actions:
			raise Exception('Output shape of model should be (nb_samples, nb_actions).')

	def get_game_data(self, game):
		frame = game.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None

	def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False, observe=0, checkpoint=None):
		print("Training:")
		loss_history = []
		reward_history = []
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon
		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0
		for epoch in range(nb_epoch):
			loss = 0.
			game.reset()
			self.clear_frames()
			if reset_memory:
				self.reset_memory()
			game_over = False
			S = self.get_game_data(game)
			while not game_over:
				if np.random.random() < epsilon or epoch < observe:
					a = int(np.random.randint(game.nb_actions))
				else:
					q = model.predict(S.reshape(1, 1, 120, 160))
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
					model.save_weights('weights.dat')
			win_count = game.is_won()
			if epsilon > final_epsilon and epoch >= observe:
				epsilon -= delta
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Total Reward {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))
			reward_history.append(game.is_won())
			loss_history.append(game.is_won())
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

	def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
		self.check_game_compatibility(game)
		model = self.model
		win_count = 0
		frames = []
		for epoch in range(nb_epoch):
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			if visualize:
				frames.append(game.draw())
			game_over = False
			while not game_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, game.nb_actions))
				else:
					q = model.predict(S)[0]
					possible_actions = game.get_possible_actions()
					q = [q[i] for i in possible_actions]
					action = possible_actions[np.argmax(q)]
				game.play(action)
				S = self.get_game_data(game)
				if visualize:
					frames.append(game.draw())
				game_over = game.is_over()
			if game.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))
		if visualize:
			if 'images' not in os.listdir('.'):
				os.mkdir('images')
			for i in range(len(frames)):
				plt.imshow(frames[i], interpolation='none')
				plt.savefig("images/" + game.name + str(i) + ".png")
