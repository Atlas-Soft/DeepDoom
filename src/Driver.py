from qlearning4k import Agent
from DoomGame import DGame
from Models import PolicyModel

model = PolicyModel()

doom = DGame('configs/basic.cfg')

agent = Agent(model=model.model, memory_size=-1, nb_frames=1)
agent.train(doom, batch_size=75, nb_epoch=1000, gamma=0.8)

model.save_weights("w_0.h5")
