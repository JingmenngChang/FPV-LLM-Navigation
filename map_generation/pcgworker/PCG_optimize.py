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


def main():

    # create worker
    PCGWorker_ = PCGWorker(9,9)

    # generate seed and render
    seed = PCGWorker_.generate()
    PCGWorker_.render(seed,wind_name = "canvas",write_ = False,write_id = 0,output = False,verbose = False)
    cv2.waitKey(0)

    # optimize seed and render
    seed2 = PCGWorker_.optimize(wave = seed,max_iter = 800)
    PCGWorker_.render(seed2,wind_name = "canvas",write_ = False,write_id = 1,output = False,verbose = False)
    _,_ = PCGWorker_.connectivity_analysis(seed2)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
