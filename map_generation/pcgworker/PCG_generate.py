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
    start = time.time()
    seed = PCGWorker_.generate()
    end = time.time() - start

    PCGWorker_.render(seed,wind_name = "canvas",write_ = False,write_id = 0,output = False,verbose = False)
    print("Time to generate: " + str(end))
    cv2.waitKey(0)

    # write wave to json file
    PCGWorker_.to_file(seed,filename = "wave.json")

    _,_ = PCGWorker_.connectivity_analysis(seed)

if __name__ == "__main__":
    main()
