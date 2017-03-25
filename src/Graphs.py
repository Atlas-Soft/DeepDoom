import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    for filename in sorted(os.listdir('../data/results/')):
        data = np.loadtxt('../data/results/' + filename)
        avg_ = data[:,1]
        min_ = data[:,2]
        max_ = data[:,3]
        plt.plot(min_, color='#e6e6e6')
        plt.plot(max_, color='#e6e6e6')
        plt.fill_between(list(range(len(avg_))), min_,max_,interpolate=True,color='#e6e6e6')
        plt.plot(avg_, color='blue')
        plt.title('Average Reward on ' + filename[:-13])
        plt.ylabel('Average Reward Per Epoch')
        plt.xlabel('Training Epochs')
        plt.savefig("../doc/figures/" + filename[:-13] + "_training_results.png")
        plt.figure()
