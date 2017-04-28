import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    for filename in sorted(os.listdir('../data/results/')):
        labels = filename[:-4].split('_')
        data = np.loadtxt('../data/results/' + filename)
        avg_ = data[:,1]
        min_ = data[:,2]
        max_ = data[:,3]
        plt.plot(min_, color='#e6e6e6')
        plt.plot(max_, color='#e6e6e6')
        plt.fill_between(list(range(len(avg_))), min_,max_,interpolate=True,color='#e6e6e6')
        plt.plot(avg_, color='blue')
        if filename == 'double-dqlearn_HDQNModel_all-skills.csv'
        plt.title('Training Reward, Algo: ' + labels[0] + ', Model: '  + labels[1] + ', Config: ' + labels[2], fontsize=10)
        plt.ylabel('Average Reward Per Epoch')
        plt.xlabel('Training Epochs')
        plt.savefig("../doc/figures/" + filename[:-4] + "_training_results.png")
        plt.figure()

    data_0 = np.loadtxt('../data/results/double-dqlearn_DQNModel_all-skills.csv')
    data_1 = np.loadtxt('../data/results/double-dqlearn_HDQNModel_all-skills.csv')
    avg_ = data_1[:,1]
    min_ = data_1[:,2]
    max_ = data_1[:,3]
    plt.plot(min_, color='#e6e6e6')
    plt.plot(max_, color='#e6e6e6')
    plt.fill_between(list(range(len(avg_))), min_,max_,interpolate=True,color='#e6e6e6')
    line1, = plt.plot(avg_, color='blue', label='h-DQN')
    line2, = plt.plot(data_0[:,1], color='red', label='DQN')
    plt.legend(handles=[line1, line2])
    plt.title('Training Reward, DQN vs h-DQN, Config: All-Skills', fontsize=10)
    plt.ylabel('Average Reward Per Epoch')
    plt.xlabel('Training Epochs')
    plt.savefig("../doc/figures/all_skills_HDQNvsDQN_training_results.png")
    plt.figure()
