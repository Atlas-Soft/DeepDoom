from Qlearning4k import QLearn
from Doom import Doom
from Models import PolicyModel

model = PolicyModel()

doom = Doom('configs/basic.cfg', frame_tics=4)

qlearn = QLearn(model=model.model, memory_size=-1)
qlearn.train(doom, batch_size=100, nb_epoch=1000, gamma=0.9, checkpoint=100, filename='w_0.h5')

model.save_weights("w_0.h5")
