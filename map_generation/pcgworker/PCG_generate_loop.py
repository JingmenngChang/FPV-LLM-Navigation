import copy
import json
import cv2
from cv2 import mean
import numpy as np
import math
import time
from colorama import Fore, Back, Style
from colorama import init
import matplotlib.pyplot as plt

import threading
import multiprocessing

from PCGWorker import *

# compute the Shannon entropy of a given array
def shannon_entropy_of_wavelet(array_):

        entropy = 0
        sum_weight = sum(array_)
        probabilities = []
        for i in array_:
            probabilities.append(i/sum_weight)

        sum_weight = sum(probabilities)
        probabilities = [x/sum_weight for x in probabilities]
        probabilities_arr = np.array(probabilities)

        entropy = -probabilities_arr.dot(np.log(probabilities))

        return entropy

# get tile distribution of a wave
def get_tile_distribution(wave_ = Wave(9,9)):
    tile_distribution = [0.0001,0.0001,0.0001,0.0001,0.0001]
    for i in range(len(wave_.wave_oriented)):
        if wave_.wave_oriented[i][0][0]>=1 and wave_.wave_oriented[i][0][0]<=5:
            tile_distribution[wave_.wave_oriented[i][0][0]-1] += 1

    return tile_distribution

def main():

    # create worker
    PCGWorker_ = PCGWorker(9,9)

    entropy_distribution = []
    for i in range(200):
        entropy_distribution.append(0)

    while True:
        # generate seed and render
        start = time.time()
        seed = PCGWorker_.generate(per_step_debug=False)
        end = time.time() - start

        PCGWorker_.render(seed,wind_name = "canvas",write_ = False,write_id = 0,output = False,verbose = False)
        print("Time to generate: " + str(end))
        cv2.waitKey(1)

        # tile_dist = get_tile_distribution(seed)
        # ent_ = shannon_entropy_of_wavelet(tile_dist)
        # ent_id = int(ent_*100)
        # ent_id = min(ent_id,199)
        # ent_id = max(ent_id,0)
        # entropy_distribution[ent_id] += 1
        # # draw tile_dist with bar graph, non-blocking
        # plt.bar(range(len(entropy_distribution)),entropy_distribution)
        # plt.show(block=False)
        # plt.pause(0.001)
        # plt.cla()
        # # compute mean entropy
        # mean_entropy = 0
        # e_num = 0
        # for i in range(len(entropy_distribution)):
        #     if entropy_distribution[i] != 0:
        #         mean_entropy += i*entropy_distribution[i]
        #         e_num += entropy_distribution[i]
        # mean_entropy = mean_entropy/e_num
        # print("Mean entropy: " + str(mean_entropy))


if __name__ == "__main__":
    main()
