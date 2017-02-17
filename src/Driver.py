import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Qlearning4k import QLearnAgent
from Doom import Doom
from Models import PolicyModel


def train():
    '''
    '''
    model = PolicyModel()

    doom = Doom('configs/basic.cfg', frame_tics=5)

    agent = QLearnAgent(model=model, memory_size=10000, nb_frames=1)
    agent.train(doom, batch_size=10, nb_epoch=100, steps=5000, alpha=0.01, gamma=0.99, observe = 10, epsilon=[1., .1], epsilon_rate=0.25, checkpoint=5, filename='basic_0.h5')

    model.save_weights("basic_0.h5")

def play():
    '''
    '''
    model = PolicyModel()
    model.load_weights("basic_0.h5")

    doom = Doom('configs/basic.cfg', frame_tics=5)

    agent = QLearnAgent(model=model, nb_frames=4)

    doom.run(agent, save_replay='basic.lmp')

    doom.replay('basic.lmp')


def test():

    doom = Doom('configs/curved_turning.cfg', frame_tics=5)
    doom.human_play()

if __name__ == '__main__':
    train()
    #test()
    #play()
