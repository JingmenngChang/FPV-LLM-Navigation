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

from pandas import wide_to_long

try:
    from .c_acc import propagate2
except ImportError:
    try:
        from c_acc import propagate2
    except ImportError:
        print("Warning: c_acc module not found. Some functions may not work properly.")
        def propagate2(*args, **kwargs):
            raise ImportError("c_acc module is not available")

###############################################################################
# PCG by wave function collapse
###############################################################################

# Wave
class Wave:
    def __init__(self, width = 9, height = 9):
        self.entropy = []
        self.wave_oriented = []
        self.oriented_weight_dict = []
        self.width = width
        self.height = height
        self.mask = []
        self.wave_collapsed= []
        for i in range(self.width * self.height):
            d = []
            d.append((1,0))
            self.wave_collapsed.append(d)

    def clear(self):
        self.entropy.clear()
        self.wave_oriented.clear()
        self.oriented_weight_dict.clear()
        self.mask.clear()
        self.wave_collapsed.clear()
        for i in range(self.width * self.height):
            d = []
            d.append((1,0))
            self.wave_collapsed.append(d)

    def apply_result(self):
        self.wave_collapsed = copy.deepcopy(self.wave_oriented)

    def get_result(self):
        return self.wave_collapsed, (len(self.wave_collapsed)!=0)
        
# PCGWorker
class PCGWorker:

    # __init__
    def __init__(self, width, height, num_processes = 0):
        self.Width = width
        self.Height = height
        self.num_processes = num_processes

        # 1-5 bricks
        # 6-9 corners
        # 10-13 ramps
        self.candidates = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        self.candidates_X_symmetry = {1,2,3,4,5}
        self.candidates_L_symmetry = {6,7,8,9,10,11,12,13}
        self.candidates_weight = [1,1,1,1,1,1,1,1,1,1,1,1,1]

        self.oriented_candidates = []
        self.oriented_candidates_weight = []
        # weight dictionary for each cell
        self.oriented_candidates_weight_dict = {}

        # H for horizontal
        # V for vertical
        # Rotation: 0->0 , 1->90, 2->180, 3->270
        self.rules_handcraft = {
                (1,0,2,0,'H'),
                (2,0,3,0,'H'),
                (3,0,4,0,'H'),
                (4,0,5,0,'H'),
                (5,0,4,0,'H'),
                (4,0,3,0,'H'),
                (3,0,2,0,'H'),
                (2,0,1,0,'H'),
                (1,0,1,0,'H'),
                (2,0,2,0,'H'),
                (3,0,3,0,'H'),
                (4,0,4,0,'H'),
                (5,0,5,0,'H'),

                # # new rules for bricks
                # (1,0,3,0,'H'),
                # (1,0,4,0,'H'),
                # (1,0,5,0,'H'),
                # (2,0,4,0,'H'),
                # (2,0,5,0,'H'),
                # (3,0,5,0,'H'),

                (6,0,1,0,'H'),
                (2,0,6,0,'H'),
                (1,0,6,0,'V'),
                (6,0,2,0,'V'),

                (7,0,2,0,'H'),
                (3,0,7,0,'H'),
                (2,0,7,0,'V'),
                (7,0,3,0,'V'),

                (8,0,3,0,'H'),
                (4,0,8,0,'H'),
                (3,0,8,0,'V'),
                (8,0,4,0,'V'),

                (9,0,4,0,'H'),
                (5,0,9,0,'H'),
                (4,0,9,0,'V'),
                (9,0,5,0,'V'),

                (2,0,10,0,'V'),
                (10,0,1,0,'V'),
                (1,0,10,0,'H'),
                (10,0,10,0,'H'),
                (10,0,1,0,'H'),

                (3,0,11,0,'V'),
                (11,0,2,0,'V'),
                (2,0,11,0,'H'),
                (11,0,11,0,'H'),
                (11,0,2,0,'H'),

                (4,0,12,0,'V'),
                (12,0,3,0,'V'),
                (3,0,12,0,'H'),
                (12,0,12,0,'H'),
                (12,0,3,0,'H'),

                (5,0,13,0,'V'),
                (13,0,4,0,'V'),
                (4,0,13,0,'H'),
                (13,0,13,0,'H'),
                (13,0,4,0,'H')

                # new rules for ramps
            }
        self.rules = set()
        self.rules_h = set()
        self.rules_v = set()

        np.random.seed(int(time.time()))

        # rotate candidates
        self.oriented_candidates.clear()
        self.oriented_candidates_weight.clear()

        for i in range(len(self.candidates)):
            candidate_0 = (self.candidates[i],0)
            candidate_90 = self.rotate_clockwise(candidate_0)
            candidate_180 = self.rotate_clockwise(candidate_90)
            candidate_270 = self.rotate_clockwise(candidate_180)
            self.oriented_candidates.append(candidate_0)
            self.oriented_candidates.append(candidate_90)
            self.oriented_candidates.append(candidate_180)
            self.oriented_candidates.append(candidate_270)
            self.oriented_candidates_weight.append(self.candidates_weight[i])
            self.oriented_candidates_weight.append(self.candidates_weight[i])
            self.oriented_candidates_weight.append(self.candidates_weight[i])
            self.oriented_candidates_weight.append(self.candidates_weight[i])
            self.oriented_candidates_weight_dict[candidate_0] = self.candidates_weight[i]
            self.oriented_candidates_weight_dict[candidate_90] = self.candidates_weight[i]
            self.oriented_candidates_weight_dict[candidate_180] = self.candidates_weight[i]
            self.oriented_candidates_weight_dict[candidate_270] = self.candidates_weight[i]

        self.rules = self.symmetry_generator(self.rules_handcraft)
        self.rules_h, self.rules_v = self.separate_rules(self.rules)


    # initialize wave
    def build_wave(self):
        wave = Wave(9,9)
        for j in range(self.Height * self.Width):
            wave.wave_oriented.append(self.oriented_candidates.copy())
            wave.entropy.append(1)
            wave.oriented_weight_dict.append(self.oriented_candidates_weight_dict.copy())
        return wave
    
    def build_wave_conditional(self, wave_old = Wave(9,9),new_weight = 81):
        wave = Wave(9,9)
        for j in range(self.Height * self.Width):
            wave.wave_oriented.append(self.oriented_candidates.copy())
            wave.entropy.append(1)
            wave.oriented_weight_dict.append(self.oriented_candidates_weight_dict.copy())
        for j in range(self.Height):
            for i in range(self.Width):
                c = wave_old.wave_oriented[j*self.Width+i][0]
                wave.oriented_weight_dict[j*self.Width+i] = self.oriented_candidates_weight_dict.copy()
                wave.oriented_weight_dict[j*self.Width+i][c] = new_weight
                
        return wave
        
    def rotate_clockwise(self,tp = (0,0)):

        if tp[0] in self.candidates_X_symmetry:
            return tp
        else:
            if tp[1] <=2:
                tp1 = (tp[0],tp[1]+1)
                return tp1
            elif tp[1] == 3:
                tp1 = (tp[0],0)
                return tp1
            else:
                return tp

    # symmetry generator
    def symmetry_generator(self,rules2):

        # print(rules2)

        rules_ = set()

        # for each rule in rules2, rotate clockwise 4 times
        for r in rules2:

            cand1 = (r[0],r[1])
            cand2 = (r[2],r[3])

            # rot 0
            rule_0 = r

            # rot 90
            rot_0_1 = self.rotate_clockwise(cand1)
            rot_0_2 = self.rotate_clockwise(cand2)
            if r[4] == 'H':
                rule_90 = (rot_0_1[0],rot_0_1[1],rot_0_2[0],rot_0_2[1],'V')
            else:
                rule_90 = (rot_0_2[0],rot_0_2[1],rot_0_1[0],rot_0_1[1],'H')

            # rot 180
            rot_90_1 = self.rotate_clockwise(self.rotate_clockwise(cand1))
            rot_90_2 = self.rotate_clockwise(self.rotate_clockwise(cand2))
            if r[4] == 'H':
                rule_180 = (rot_90_2[0],rot_90_2[1],rot_90_1[0],rot_90_1[1],'H')
            else:
                rule_180 = (rot_90_2[0],rot_90_2[1],rot_90_1[0],rot_90_1[1],'V')

            # rot 270
            rot_270_1 = self.rotate_clockwise(self.rotate_clockwise(self.rotate_clockwise(cand1)))
            rot_270_2 = self.rotate_clockwise(self.rotate_clockwise(self.rotate_clockwise(cand2)))
            if r[4] == 'H':
                rule_270 = (rot_270_2[0],rot_270_2[1],rot_270_1[0],rot_270_1[1],'V')
            else:
                rule_270 = (rot_270_1[0],rot_270_1[1],rot_270_2[0],rot_270_2[1],'H')

            # add to rules
            rules_.add(rule_0)
            rules_.add(rule_90)
            rules_.add(rule_180)
            rules_.add(rule_270)

        # print(rules_)
        return rules_

    def separate_rules(self, rules):
        rules_h = set()
        rules_v = set()

        for r in rules:
            if r[4] == 'H':
                rules_h.add((r[0],r[1],r[2],r[3]))
            else:
                rules_v.add((r[0],r[1],r[2],r[3]))

        return rules_h,rules_v

    # collapse state detection
    def collapse_detection(self, wave_index, wave = Wave(9,9)):
        if len(wave.wave_oriented[wave_index]) != 1:
            return False
        else:
            return True

    # get weight through candidate
    def get_weight_of_candidate(self, index, candidate, wave = Wave(9,9)):
        return wave.oriented_weight_dict[index][candidate]

    def distributional_trigger(self, wavelet_index,wave = Wave(9,9)):

        collapse_element_index = 0

        # construct a new weight array for the wavelet
        weight_array = []
        for i in range(len(wave.wave_oriented[wavelet_index])):
            weight_array.append(self.get_weight_of_candidate(wavelet_index,wave.wave_oriented[wavelet_index][i],wave))

        # generate a random number in [0, sum(weight_array)]
        random_number = np.random.rand() * sum(weight_array)

        # find the collapse element
        for i in range(len(weight_array)):
            random_number -= weight_array[i]
            if random_number <= 0:
                collapse_element_index = i
                break

        return collapse_element_index

    # calculate the Shannon entropy of a wavelet
    def shannon_entropy_of_wavelet(self, index, wavelet,wave = Wave(9,9)):

        entropy = 0
        sum_weight = 0
        probabilities = []
        for i in wavelet:
            probabilities.append(self.get_weight_of_candidate(index,i,wave))

        sum_weight = sum(probabilities)
        probabilities = [x/sum_weight for x in probabilities]
        probabilities_arr = np.array(probabilities)

        entropy = -probabilities_arr.dot(np.log(probabilities))

        entropy += np.random.rand() * 0.001

        return entropy

    # update entropy
    def update_wave_entropy(self,wave = Wave(9,9)):
        for i in range(len(wave.wave_oriented)):
            wave.entropy[i] = self.shannon_entropy_of_wavelet(i,wave.wave_oriented[i],wave)

    # find an element to collapse
    def find_minimum_entropy_position(self,wave = Wave(9,9)):
        # find minimum entropy index of entropy
        min_entropy_index = 0
        min_entropy = 1e10
        for i in range(len(wave.entropy)):
            if wave.entropy[i] < min_entropy and self.collapse_detection(i,wave) == False:
                min_entropy = wave.entropy[i]
                min_entropy_index = i

        return min_entropy_index

    # propagate the wave function
    # active-propagation
    def propagate(self,wave = Wave(9,9)):
        
        # neighbors :
        #         (i,j-1)
        # (i-1,j) (i,j  ) (i+1,j)
        #         (i,j+1)
        # for k in range(2):

        propagate2(self.rules_h,self.rules_v,wave.wave_oriented,self.Width,self.Height)

        self.update_wave_entropy(wave)

    # collapse the wavelet
    def collapse_at_position(self,wavelet_index,wave = Wave(9,9)):
        
        collapse_element_index = self.distributional_trigger(wavelet_index,wave)
        wavelet = []
        wavelet.append(wave.wave_oriented[wavelet_index][collapse_element_index])
        wave.wave_oriented[wavelet_index].clear()
        # deep copy
        wave.wave_oriented[wavelet_index] = copy.deepcopy(wavelet)
        wave.entropy[wavelet_index] = 0
        # wave.wave_oriented[wavelet_index] = wavelet.copy()

    # check if the wave is collapsed
    def is_collapsed(self,wave = Wave(9,9)):
        for i in range(len(wave.wave_oriented)):
            if self.collapse_detection(i,wave) == False:
                return False
        return True

    # contradiction check
    def contradiction_check(self,wave = Wave(9,9)):
        for j in range(self.Height):
            for i in range(self.Width):

                if self.collapse_detection(j*self.Width+i,wave) == False:
                    print(j*self.Width+i)
                    continue

                i_0 = i-1
                j_0 = j-1
                i_1 = i+1
                j_1 = j+1

                # construct local context and compare it with rules
                # (i-1,j)
                if i_0 >= 0:
                    for element_i in wave.wave_oriented[j*self.Width+i]:
                        for element_j in wave.wave_oriented[j*self.Width+i_0]:
                            if not (element_j[0],element_j[1],element_i[0],element_i[1],'H') in self.rules:
                                return False
                # (i+1,j)
                if i_1 < self.Width:
                    for element_i in wave.wave_oriented[j*self.Width+i]:
                        for element_j in wave.wave_oriented[j*self.Width+i_1]:
                            if not (element_i[0],element_i[1],element_j[0],element_j[1],'H') in self.rules:
                                return False
                # (i,j-1)
                if j_0 >= 0:
                    for element_i in wave.wave_oriented[j*self.Width+i]:
                        for element_j in wave.wave_oriented[j_0*self.Width+i]:
                            if not (element_j[0],element_j[1],element_i[0],element_i[1],'V') in self.rules:
                                return False
                # (i,j+1)
                if j_1 < self.Height:
                    for element_i in wave.wave_oriented[j*self.Width+i]:
                        for element_j in wave.wave_oriented[j_1*self.Width+i]:
                            if not (element_i[0],element_i[1],element_j[0],element_j[1],'V') in self.rules:
                                return False
        return True

    # render
    def render(self, wave = Wave(9,9), wind_name = "canvas",write_ = False,write_id = 0,output = False, verbose = False, border = True):

        # bricks
        tile1 = np.zeros((20,20,3), np.uint8)
        tile2 = np.zeros((20,20,3), np.uint8)
        tile2[:,:] = (50,50,50)
        tile3 = np.zeros((20,20,3), np.uint8)
        tile3[:,:] = (100,100,100)
        tile4 = np.zeros((20,20,3), np.uint8)
        tile4[:,:] = (150,150,150)
        tile5 = np.zeros((20,20,3), np.uint8)
        tile5[:,:] = (200,200,200)

        # corners
        tile6 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                if j<i:
                    tile6[j,i] = (0,0,0)
                else:
                    tile6[j,i] = (50,50,50)
        tile7 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                if j<i:
                    tile7[j,i] = (50,50,50)
                else:
                    tile7[j,i] = (100,100,100)
        tile8 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                if j<i:
                    tile8[j,i] = (100,100,100)
                else:
                    tile8[j,i] = (150,150,150)
        tile9 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                if j<i:
                    tile9[j,i] = (150,150,150)
                else:
                    tile9[j,i] = (200,200,200)

        # ramps
        tile10 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                tile10[j,i] = (50-j*2.5,50-j*2.5,50-j*2.5)
        tile11 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                tile11[j,i] = (100-j*2.5,100-j*2.5,100-j*2.5)
        tile12 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                tile12[j,i] = (150-j*2.5,150-j*2.5,150-j*2.5)
        tile13 = np.zeros((20,20,3), np.uint8)
        for j in range(20):
            for i in range(20):
                tile13[j,i] = (200-j*2.5,200-j*2.5,200-j*2.5)

        tiles = [] 
        tiles.append(tile1)
        tiles.append(tile2)
        tiles.append(tile3)
        tiles.append(tile4)
        tiles.append(tile5)
        tiles.append(tile6)
        tiles.append(tile7)
        tiles.append(tile8)
        tiles.append(tile9)
        tiles.append(tile10)
        tiles.append(tile11)
        tiles.append(tile12)
        tiles.append(tile13)
        # draw a black rectangle on the border of each tile
        if border:
            for i in range(len(tiles)):
                tiles[i][0,:] = 0
                tiles[i][:,0] = 0
                tiles[i][19,:] = 0
                tiles[i][:,19] = 0

        # conbine Height x Width tiles into a single image
        canvas = np.zeros((self.Height*20,self.Width*20,3), np.uint8)

        for j in range(self.Height):
            for i in range(self.Width):
                if self.collapse_detection(j*self.Width+i,wave) == True:
                    if wave.wave_oriented[j*self.Width+i][0][1] == 1:
                        # rotate the tile 90 degrees clockwise
                        canvas[j*20:(j+1)*20,i*20:(i+1)*20] = np.rot90(tiles[wave.wave_oriented[j*self.Width+i][0][0]-1],-1).copy()
                    elif wave.wave_oriented[j*self.Width+i][0][1] == 2:
                        # rotate the tile 180 degrees clockwise
                        canvas[j*20:(j+1)*20,i*20:(i+1)*20] = np.rot90(tiles[wave.wave_oriented[j*self.Width+i][0][0]-1],-2).copy()
                    elif wave.wave_oriented[j*self.Width+i][0][1] == 3:
                        # rotate the tile 270 degrees clockwise
                        canvas[j*20:(j+1)*20,i*20:(i+1)*20] = np.rot90(tiles[wave.wave_oriented[j*self.Width+i][0][0]-1],-3).copy()
                    else:
                        canvas[j*20:(j+1)*20,i*20:(i+1)*20] = tiles[wave.wave_oriented[j*self.Width+i][0][0]-1].copy()

        if verbose:
            for j in range(self.Height):
                for i in range(self.Width):
                    sum_ = len(wave.wave_oriented[j*self.Width+i])==1 and wave.wave_oriented[j*self.Width+i][0][0] or len(wave.wave_oriented[j*self.Width+i])
                    dir_ = wave.wave_oriented[j*self.Width+i][0][1]
                    mean = sum_
                    mean_str = str(round(mean,2)) + ":" + str(dir_)
                    # using str.format() to format the string
                    mean_str = "{:>5}".format(mean_str)
                    if len(wave.wave_oriented[j*self.Width+i]) == len(self.oriented_candidates):
                        print(Style.RESET_ALL + Fore.CYAN + mean_str + "," + Fore.LIGHTBLUE_EX, end=' ')
                    elif len(wave.wave_oriented[j*self.Width+i]) == 1:
                        print(Style.RESET_ALL + Fore.GREEN + mean_str + ",", end=' ')
                    else:
                        print(Style.RESET_ALL + Fore.RED  + mean_str + ",", end=' ')
                print()

        if write_ == True:
            id_string = str(write_id)
            cv2.imwrite(id_string + ".png", canvas)
            # save wave to a json file
            with open(id_string + ".json", 'w') as outfile:
                json.dump(wave.wave_oriented, outfile)
            
        if not output:
            cv2.imshow(wind_name,canvas)
            cv2.waitKey(1)
        else:
            return canvas

    def super_render(self,wave1,wave2,wave3):

        cav1 = self.render(wave1,output=True)
        cav2 = self.render(wave2,output=True)
        cav3 = self.render(wave3,output=True)

        # make a green border for all three images
        
        cav1[0,:] = (0,255,0)
        cav1[:,0] = (0,255,0)
        cav1[self.Height*20-1,:] = (0,255,0)
        cav1[:,self.Width*20-1] = (0,255,0)

        cav2[0,:] = (0,255,0)
        cav2[:,0] = (0,255,0)
        cav2[self.Height*20-1,:] = (0,255,0)
        cav2[:,self.Width*20-1] = (0,255,0)

        cav3[0,:] = (0,255,0)
        cav3[:,0] = (0,255,0)
        cav3[self.Height*20-1,:] = (0,255,0)
        cav3[:,self.Width*20-1] = (0,255,0)


        # combine the three cavases into one horizontal canvas
        canvas = np.hstack((cav1,cav2,cav3))

        cv2.imshow("canvas_all",canvas)
        cv2.waitKey(1)

        return canvas

    # fitness
    def fitness(self,wave = Wave(9,9)):
            
        # calculate second order dirivative along x axis and y axis
        d1x = np.zeros((self.Height,self.Width), np.float32)
        d1y = np.zeros((self.Height,self.Width), np.float32)
        d2x = np.zeros((self.Height,self.Width), np.float32)
        d2y = np.zeros((self.Height,self.Width), np.float32)

        # get d1
        for j in range(self.Height):
            for i in range(self.Width):
                i_0 = i-1
                j_0 = j-1
                if i_0 >= 0:
                    w1 = wave.wave_oriented[j*self.Width+i][0][0]
                    w2 = wave.wave_oriented[j*self.Width+i_0][0][0]
                    if w1 in range(6,10):
                        w1 -= 5
                    elif w1 in range(10,14):
                        w1 -= 8.5
                    if w2 in range(6,10):
                        w2 -= 5
                    elif w2 in range(10,14):
                        w2 -= 8.5
                    d1x[j,i_0] = w1 - w2
                if j_0 >= 0:
                    w1 = wave.wave_oriented[j_0*self.Width+i][0][0]
                    w2 = wave.wave_oriented[j*self.Width+i][0][0]
                    if w1 in range(6,10):
                        w1 -= 5
                    elif w1 in range(10,14):
                        w1 -= 8.5
                    if w2 in range(6,10):
                        w2 -= 5
                    elif w2 in range(10,14):
                        w2 -= 8.5
                    d1y[j_0,i] = w1 - w2
        # get d2
        for j in range(self.Height):
            for i in range(self.Width):
                i_0 = i-1
                j_0 = j-1
                if i_0 >= 0:
                    d2x[j,i_0] = d1x[j,i] - d1x[j,i_0]
                if j_0 >= 0:
                    d2y[j_0,i] = d1y[j,i] - d1y[j_0,i]

        # compute sum of absolute value of d2
        smoothness = 0
        for j in range(self.Height):
            for i in range(self.Width):
                smoothness += abs(d2x[j,i]) + abs(d2y[j,i])

        dir_x = [0.0001,0.0001]
        dir_y = [0.0001,0.0001]
        for j in range(self.Height-1):
            for i in range(self.Width-1):
                if d1x[j,i] > 0:
                    dir_x[0] += 1
                elif d1x[j,i] < 0:
                    dir_x[1] += 1
                # else:
                #     dir_x[2] += 1
                if d1y[j,i] > 0:
                    dir_y[0] += 1
                elif d1y[j,i] < 0:
                    dir_y[1] += 1
                # else:
                #     dir_y[2] += 1

        # compute shannon entropy of dir_x and dir_y
        entropy_x = 0
        entropy_y = 0
        sum_x = 0.001
        sum_y = 0.001
        for i in range(len(dir_x)):
            sum_x += dir_x[i]
            sum_y += dir_y[i]
        for i in range(len(dir_x)):
            if sum_x != 0:
                entropy_x += -dir_x[i]/sum_x * math.log(dir_x[i]/sum_x)
            if sum_y != 0:
                entropy_y += -dir_y[i]/sum_y * math.log(dir_y[i]/sum_y)

        # compute balanceness of element distribution
        balance = 0.0001
        candidates_n = self.candidates_weight.copy()
        for i in range(len(candidates_n)):
            candidates_n[i] = 0
        for j in range(self.Height):
            for i in range(self.Width):
                w1 = wave.wave_oriented[j*self.Width+i][0][0]
                if w1 in range(6,10):
                    candidates_n[w1-4-1] += 0.5
                    candidates_n[w1-5-1] += 0.5
                elif w1 in range(10,14):
                    candidates_n[w1-8-1] += 0.5
                    candidates_n[w1-9-1] += 0.5
                else:
                    candidates_n[w1-1] += 1
                    
        # compute average of candidates_n
        s_ = sum(candidates_n)
        for i in range(len(candidates_n)):
            candidates_n[i]/=s_
        avg_ = sum(candidates_n)/len(candidates_n)
        for i in range(len(candidates_n)):
            balance += abs(candidates_n[i]-avg_)

        # count concave corner
        concave_corners = 0
        for j in range(self.Height):
            for i in range(self.Width):
                w = wave.wave_oriented[j*self.Width+i][0][0]
                if w in range(1,6):
                    d0 = (i+1,j)
                    d1 = (i,j+1)
                    d2 = (i-1,j)
                    d3 = (i,j-1)
                    w0 = 0
                    w1 = 0
                    w2 = 0
                    w3 = 0
                    if d0[0] < self.Width:
                        w0 = wave.wave_oriented[d0[1]*self.Width+d0[0]][0][0]
                        if not w0 in range(1,6):
                            w0 = 0
                    if d1[1] < self.Height:
                        w1 = wave.wave_oriented[d1[1]*self.Width+d1[0]][0][0]
                        if not w1 in range(1,6):
                            w1 = 0
                    if d2[0] >= 0:
                        w2 = wave.wave_oriented[d2[1]*self.Width+d2[0]][0][0]
                        if not w2 in range(1,6):
                            w2 = 0
                    if d3[1] >= 0:
                        w3 = wave.wave_oriented[d3[1]*self.Width+d3[0]][0][0]
                        if not w3 in range(1,6):
                            w3 = 0
                    if w0 > w and w1 >w and w2 <= w and w3 <= w:
                        concave_corners += 1
                    if w1 > w and w2 >w and w3 <= w and w0 <= w:
                        concave_corners += 1
                    if w2 > w and w3 >w and w0 <= w and w1 <= w:
                        concave_corners += 1
                    if w3 > w and w0 >w and w1 <= w and w2 <= w:
                        concave_corners += 1
                    

        fitness_ =  7 * (abs(entropy_x) + abs(entropy_y)) + smoothness + 50 * balance + 2.2 * concave_corners # - 5 * abs(min_-max_)

        # print fitness
        print("fitness:",7 * (abs(entropy_x) + abs(entropy_y)),smoothness,10 * balance,concave_corners)

        # add a small purturbation to fitness
        fitness_ += np.random.rand() * 1

        return fitness_

    def clear_screen(self):
        print('\033c',end='')

    # generate
    def generate(self, per_step_debug = False):

        while True:

            # start = time.time()
            
            wave = Wave(9,9)
            wave = self.build_wave()

            self.update_wave_entropy(wave)

            # time1 = time.time() - start
            # start = time.time()
            # time2 = 0
            # time3 = 0
            # time4 = 0
            # time5 = 0

            while not self.is_collapsed(wave):

                # start = time.time()
                # 1. select the minimum entropy wavelet 
                collapse_element = self.find_minimum_entropy_position(wave)
                # time2 += time.time() - start
                # start = time.time()

                # 2. collapse the minimum entropy wavelet
                self.collapse_at_position(collapse_element,wave)
                # time3 += time.time() - start
                # start = time.time()

                if per_step_debug:
                    print('\033c',end='')
                    self.render(wave,verbose = True)
                    print("------------------------before propagate------------------------")

                # 3. propagate the wave function
                
                self.propagate(wave)
                # time4 += time.time() - start
                # start = time.time()

                if per_step_debug:
                    # print('\033c',end='')
                    self.render(wave,verbose = True)
                    print("------------------------after propagate------------------------")
                    for i in range(self.Height):
                        for j in range(self.Width):
                            entropy_str = str(round(wave.entropy[i*self.Width+j],2))
                            # using str.format() to format the string
                            print(Style.RESET_ALL + Fore.RED + "{:>5}".format(entropy_str),end='')
                        print("")
                    input()

            # # print times
            # print("Time to build wave: " + str(time1))
            # print("Time to find minimum entropy: " + str(time2))
            # print("Time to collapse: " + str(time3))
            # print("Time to propagate: " + str(time4))

            # 4. contradiction check
            if self.contradiction_check(wave):
                wave.apply_result()
                return wave
            else:
                print("Contradiction found!")
                

    # mutate
    def mutate(self, wave_old = Wave(9,9),weight_=81):

        while True:
            
            wave = Wave(9,9)
            wave = self.build_wave_conditional(wave_old = wave_old, new_weight=weight_)

            self.update_wave_entropy(wave)

            while not self.is_collapsed(wave):

                # 1. select the minimum entropy wavelet 
                collapse_element = self.find_minimum_entropy_position(wave)

                # 2. collapse the minimum entropy wavelet
                self.collapse_at_position(collapse_element,wave)

                # 3. propagate the wave function
                self.propagate(wave)
                
            # 4. contradiction check
            if self.contradiction_check(wave):
                wave.apply_result()
                return wave

    # optimize
    def optimize(self, wave = Wave(9,9), max_iter = 500, fitness_report = True):

        pareto_front = Wave(9,9)
        pareto_front = copy.deepcopy(wave)
        step_ = 0
        new_weight = self.Width * self.Height
        f_min = self.fitness(pareto_front)
        energy = []

        while step_ < max_iter:
            step_ += 1
            offspring = self.mutate(pareto_front,new_weight)
            f_ = self.fitness(offspring)
            if f_ < f_min:
                pareto_front = copy.deepcopy(offspring)
                f_min = f_
            energy.append(f_min)
            self.super_render(offspring,pareto_front,wave)
            self.clear_screen()
            print("step:",step_)

        if fitness_report:
            plt.plot(energy)
            plt.show()
            cv2.waitKey(0)

        pareto_front.apply_result()
        return pareto_front

    # mutate and optimize
    def mutate_and_optimize(self, wave = Wave(9,9), max_iter = 500, fitness_report = True):
            
        new_weight = self.Width * self.Height * 0.9

        # mutate
        new_wave = self.mutate(wave,new_weight)
        # optimize
        offspring = self.optimize(new_wave,max_iter,fitness_report)

        offspring.apply_result()
        return offspring

    # connectivity analysis
    def connectivity_analysis(self, wave = Wave(9,9),visualize_ = True, to_file = True, filename = "connectivity_map.json"):
        
        # 1. generate canvas for analysis
        canvas = self.render(wave,output=True,border=False)
        canvas_bg = canvas.copy()

        # 2. seperate each layer
        layers = []
        l1 = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        l2 = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        l3 = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        l4 = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        l5 = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        kernel = np.ones((3,3),np.uint8)

        # traverse canvas, find pixels with color(0,0,0) and set color of corresponding pixels of image l1 to white
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if canvas[j,i][0] == 0 and canvas[j,i][1] == 0 and canvas[j,i][2] == 0:
                    l1[j,i] = (255,255,255)
                else:
                    l1[j,i] = (0,0,0)
        l1 = cv2.erode(l1,kernel,iterations = 1)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if canvas[j,i][0] == 50 and canvas[j,i][1] == 50 and canvas[j,i][2] == 50:
                    l2[j,i] = (255,255,255)
                else:
                    l2[j,i] = (0,0,0)
        l2 = cv2.erode(l2,kernel,iterations = 1)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if canvas[j,i][0] == 100 and canvas[j,i][1] == 100 and canvas[j,i][2] == 100:
                    l3[j,i] = (255,255,255)
                else:
                    l3[j,i] = (0,0,0)
        l3 = cv2.erode(l3,kernel,iterations = 1)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if canvas[j,i][0] == 150 and canvas[j,i][1] == 150 and canvas[j,i][2] == 150:
                    l4[j,i] = (255,255,255)
                else:
                    l4[j,i] = (0,0,0)
        l4 = cv2.erode(l4,kernel,iterations = 1)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if canvas[j,i][0] == 200 and canvas[j,i][1] == 200 and canvas[j,i][2] == 200:
                    l5[j,i] = (255,255,255)
                else:
                    l5[j,i] = (0,0,0)
        l5 = cv2.erode(l5,kernel,iterations = 1)

        # convert l1-l5 to gray scale
        l1 = cv2.cvtColor(l1,cv2.COLOR_BGR2GRAY)
        l2 = cv2.cvtColor(l2,cv2.COLOR_BGR2GRAY)
        l3 = cv2.cvtColor(l3,cv2.COLOR_BGR2GRAY)
        l4 = cv2.cvtColor(l4,cv2.COLOR_BGR2GRAY)
        l5 = cv2.cvtColor(l5,cv2.COLOR_BGR2GRAY)
        
        # 3. opencv-connectivity analysis for each layer

        # connected area analysis
        nb_components1, mask1, _, _ = cv2.connectedComponentsWithStats(l1, connectivity=8)
        nb_components2, mask2, _, _ = cv2.connectedComponentsWithStats(l2, connectivity=8)
        nb_components3, mask3, _, _ = cv2.connectedComponentsWithStats(l3, connectivity=8)
        nb_components4, mask4, _, _ = cv2.connectedComponentsWithStats(l4, connectivity=8)
        nb_components5, mask5, _, _ = cv2.connectedComponentsWithStats(l5, connectivity=8)

        nb_components = [nb_components1,nb_components2,nb_components3,nb_components4,nb_components5]
        masks = [mask1,mask2,mask3,mask4,mask5]
        cpnts_total = sum(nb_components)-5

        # re-order masks
        for m in range(len(masks)):
            stair = 0
            for i in range(0,m):
                stair += (nb_components[i]-1)
            for i in range(0,self.Height*20):
                for j in range(0,self.Width*20):
                    if masks[m][i,j] != 0:
                        masks[m][i,j] += stair

        if cpnts_total > 1:
            # fill each area with different color
            step_ = 250/cpnts_total
            for j in range(self.Height*20):
                for i in range(self.Width*20):
                    l1[j,i] = masks[0][j,i]
                    l2[j,i] = masks[1][j,i]
                    l3[j,i] = masks[2][j,i]
                    l4[j,i] = masks[3][j,i]
                    l5[j,i] = masks[4][j,i]

        # conbine all masks into a single mask
        mask = np.zeros((self.Height*20,self.Width*20), np.uint8)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                if l1[j,i] != 0:
                    mask[j,i] = l1[j,i]
                elif l2[j,i] != 0:
                    mask[j,i] = l2[j,i]
                elif l3[j,i] != 0:
                    mask[j,i] = l3[j,i]
                elif l4[j,i] != 0:
                    mask[j,i] = l4[j,i]
                elif l5[j,i] != 0:
                    mask[j,i] = l5[j,i]
                else:
                    mask[j,i] = 0

        # print("cpnts_total:",cpnts_total)
        
        # 4. compute vectors for each ramp

        # collect all ramps
        ramps = []
        for j in range(self.Height):
            for i in range(self.Width):
                if wave.wave_oriented[j*self.Width+i][0][0] >= 10 and wave.wave_oriented[j*self.Width+i][0][0] <= 13:
                    ramps.append((i,j,wave.wave_oriented[j*self.Width+i][0][1]))

        ramp_vectors = []
        # compute bounding boxes for each ramp
        for r in ramps:
            left = r[0]*20
            right = (r[0]+1)*20
            top = r[1]*20
            bottom = (r[1]+1)*20
            rot_n = (r[2]+1)*90
            arr_l = 20
            geo_center = ((left+right)/2,(top+bottom)/2)
            backward_point = (geo_center[0]+arr_l*math.cos(math.radians(rot_n)),geo_center[1]+arr_l*math.sin(math.radians(rot_n)))
            forward_point = (geo_center[0]-arr_l*math.cos(math.radians(rot_n)),geo_center[1]-arr_l*math.sin(math.radians(rot_n)))
            ramp_vectors.append((geo_center,backward_point,forward_point))
            # draw an arrow from geo_center to forward_point
            cv2.arrowedLine(canvas_bg,(int(backward_point[0]),int(backward_point[1])),(int(forward_point[0]),int(forward_point[1])),(0,0,255),2)
            # draw bounding box on canvas
            cv2.rectangle(canvas_bg,(left,top),(right,bottom),(0,255,0),2)

        # 5. merge connected areas with ramp vectors
        for rv in ramp_vectors:

            # check if rv[1] and rv[2] are inside the image
            if rv[1][0] < 0 or rv[1][0] > self.Width*20 or rv[1][1] < 0 or rv[1][1] > self.Height*20:
                continue
            if rv[2][0] < 0 or rv[2][0] > self.Width*20 or rv[2][1] < 0 or rv[2][1] > self.Height*20:
                continue

            from_id = mask[int(rv[1][1]),int(rv[1][0])]
            to_id = mask[int(rv[2][1]),int(rv[2][0])]

            vertical_ramp = False
            if abs(rv[1][0]-rv[2][0])< abs(rv[1][1]-rv[2][1]):
                vertical_ramp = True
            else:
                vertical_ramp = False
            
            # merge from_area and to_area
            for j in range(self.Height*20):
                for i in range(self.Width*20):
                    if mask[j,i] == from_id:
                        mask[j,i] = to_id
            # fill ramp area
            if vertical_ramp:
                for j in range(int(rv[0][1]-12),int(rv[0][1]+12)):
                    for i in range(int(rv[0][0]-9),int(rv[0][0]+9)):
                        mask[j,i] = to_id
            else:
                for j in range(int(rv[0][1]-9),int(rv[0][1]+9)):
                    for i in range(int(rv[0][0]-12),int(rv[0][0]+12)):
                        mask[j,i] = to_id

        # regularize the mask ID
        id_set = set()
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                id_set.add(mask[j,i])
        id_dict = {}
        id_ = 0
        for i in id_set:
            id_ += 1
            id_dict[i] = id_
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                mask[j,i] = id_dict[mask[j,i]]
        
        # mask visualization
        mask_visual = np.zeros((self.Height*20,self.Width*20,3), np.uint8)
        for j in range(self.Height*20):
            for i in range(self.Width*20):
                g = mask[j,i] * (360/len(id_set))
                # convert (g,255,255) from hsv to bgr
                if mask[j,i] != 1:
                    mask_visual[j,i] = cv2.cvtColor(np.uint8([[[g,255,255]]]),cv2.COLOR_HSV2BGR)[0,0]
                else:
                    mask_visual[j,i] = (0,0,0)

        if to_file:
            # construct a json file for mask output
            mask_json = {}
            # make mask a python list
            mask_list = []
            for j in range(self.Height*20):
                for i in range(self.Width*20):
                    mask_list.append((int)(mask[j,i]))
            
            mask_json['mask'] = mask_list.copy()
            # write json file
            with open(filename, 'w') as f:
                json.dump(mask_json, f)

            cv2.imwrite(filename[:-5]+'_mask.bmp',mask_visual)

        if visualize_:
            cv2.imshow("mask",mask_visual)
            cv2.imshow("canvas",canvas_bg)
            cv2.waitKey(0)

        wave.mask = mask.copy()
        
        return mask, mask_visual

    # serialize
    def to_file(self, wave = Wave(9,9), filename = "wave.json"):

        # construct a json file using wave
        json_file = {}
        json_file['width'] = wave.width
        json_file['height'] = wave.height

        # make wave.wave_oriented to a flat list
        wave_list = []
        for j in range(len(wave.wave_oriented)):
            for i in range(len(wave.wave_oriented[j])):
                wave_list.append(wave.wave_oriented[j][i][0])
                wave_list.append(wave.wave_oriented[j][i][1])
        json_file['data'] = copy.deepcopy(wave_list)

        # write to file
        with open(filename, 'w') as outfile:
            json.dump(json_file, outfile)

    # deserialize
    def from_file(self,filename = "wave.json"):
        
        # construct a wave from json file
        data = []
        data1 = []
        width_ = 0
        height_ = 0
        with open(filename) as json_file:
            json_file = json.load(json_file)
            data = copy.deepcopy(json_file['data'])
            width_ = json_file['width']
            height_ = json_file['height']

        wave = Wave(width_,height_)

        for i in range((int)(len(data)/2)):
            d = []
            d.append((data[i*2+0],data[i*2+1]))
            data1.append(d)

        wave.wave_oriented = copy.deepcopy(data1)
        wave.apply_result()
        
        return wave

