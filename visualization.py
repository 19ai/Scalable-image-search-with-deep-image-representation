#!/usr/bin/env python
import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time
PATH = 'output/statistics/'

# list_of_file = ["correct_prediction_before_pp.txt","correct_prediction_after_pp.txt","wrong_prediction_before_pp.txt","wrong_prediction_after_pp.txt"]

# # file = "correct_prediction_before_pp.txt"
# file = "correct_prediction_after_pp.txt"
# # file = "wrong_prediction_before_pp.txt"
# # file = "wrong_prediction_after_pp.txt"
def visualization(file, mode_1, mode_2):
    # import matplotlib.pyplot as plt
    range = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    f = open(PATH+file,"r")
    f.next()
    # if 'before' in file:
    #     x = [float(line.split()[1]) for line in f if len(line.split()) == 3]
    # elif 'after' in file:
    #     x = [float(line.split()[2]) for line in f if len(line.split()) == 3]

    if mode_1 == 'correct' and mode_2 == 'before':
        x = [float(line.split()[1]) for line in f if len(line.split()) == 7 and line.split()[6][:7] == 'correct']
    elif mode_1 == 'incorrect' and mode_2 == 'before':
        x = [float(line.split()[1]) for line in f if len(line.split()) == 7 and line.split()[6][:9] == 'incorrect']
    elif mode_1 == 'correct' and mode_2 == 'after':
        x = [float(line.split()[2]) for line in f if len(line.split()) == 7 and line.split()[6][-8:] == '-correct']
    elif mode_1 == 'incorrect' and mode_2 == 'after':
        x = [float(line.split()[2]) for line in f if len(line.split()) == 7 and line.split()[6][-9:] == 'incorrect']
    f.close()
    n, bins, patches = plt.hist(x, range, facecolor='green', edgecolor='black')
    if 'before' in file:
        plt.xlabel('Cosine similarity')
    elif 'after' in file:
        plt.xlabel('Percentage of found matching points')
    plt.ylabel('Number of predictions')
    plt.title('There are {} predictions'.format(len(x)))
    if 'before' in file:
        plt.axis([0, 1, 0, len(x)/2])
    elif 'after' in file:
        plt.axis([0, 1, 0, len(x)])
    plt.grid(True)
    plt.savefig(PATH+mode_1+'_'+mode_2+'.jpg')
    plt.clf()


if __name__ == '__main__':
    # mode_1 = correct or incorrect
    # mode_2 = before or after

    # visualization(file='general_output.txt', mode_1 = 'correct', mode_2 = 'before')
    # visualization(file='general_output.txt', mode_1 = 'incorrect', mode_2 = 'before')
    # visualization(file='general_output.txt', mode_1 = 'correct', mode_2 = 'after')
    visualization(file='general_output.txt', mode_1 = 'incorrect', mode_2 = 'after')