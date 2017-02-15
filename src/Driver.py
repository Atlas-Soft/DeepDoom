from Qlearning4k import QLearnAgent
from Doom import Doom
from Models import PolicyModel


def train():
    '''
    '''
    model = PolicyModel()

    doom = Doom('configs/basic.cfg', frame_tics=4)

    agent = QLearnAgent(model=model, memory_size=6000)
    agent.train(doom, batch_size=10, nb_epoch=3, gamma=0.9, checkpoint=100, filename='w_0.h5')

    model.save_weights("w_0.h5")

def play():
    '''
    '''
    model = PolicyModel()
    model.load_weights("w_0.h5")

    doom = Doom('configs/basic.cfg', frame_tics=4)

    agent = QLearnAgent(model=model.model)

    doom.run(agent, save_replay='basic.lmp')

    doom.replay('basic.lmp')

if __name__ == '__main__':
    train()
    play()
