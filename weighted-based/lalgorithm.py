import random
import os
import re
import sys
sys.path.append('../')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import csv
import matplotlib.pyplot as plt
import datetime
import time
import utils
import clustering_worker
import os
import pickle
from math import radians, cos, sin, asin, sqrt, degrees, atan2, ceil

def Normalization(x, y):
    query = np.array(x)
    cand = np.array(y)
    Max = np.amax([np.amax(query, axis = 0), np.amax(cand, axis = 0)], axis = 0)
    Min = np.amin([np.amin(query, axis = 0), np.amin(cand, axis = 0)], axis = 0)
        
    query[:, 0] = (query[:, 0] - Min[0])/(Max[0] - Min[0])
    cand[:, 0] = (cand[:, 0] - Min[0])/(Max[0] - Min[0])
    query[:, 1] = (query[:, 1])/180.0
    cand[:, 1] = (cand[:, 1])/180.0
    return query, cand

def distance(cand, query):
    tmp = query - cand
    for i in range(len(tmp)):
        if tmp[i][1] > 1:   #max angle discrimination is pi
            tmp[i][1] = 2-tmp[i][1]
    w =[[1.0 for i in range(2)] for j in range(len(query))]
    dist_list = tmp * w
    average_dist = sum(sum(abs(dist_list)))/(len(cand))
    return average_dist

def ExactS(traj_c, traj_q, minLen):
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    subset = {}
    N = len(traj_c)
    M = len(traj_q)
    cand, query = Normalization(traj_c, traj_q)
    for k in range(M):
#         print("stage:", k)
        for m in range(k, M):
            if m-k<minLen:
                continue
            for i in range(N):
                for j in range(i, N):
                    if j-i<minLen or j-i !=m-k:
                        continue
                    temp = distance(cand[i:j+1], query[k:m+1])
                    subset[(i, j, k, m)] = temp
                    if temp < subsim:
                        subsim = temp
                        subtraj = (i, j, k, m)
                        print(m-k)
    return subsim, subtraj, subset

def SBLCS(traj_c, traj_q, minLen, error_dict):
    cand, query = Normalization(traj_c, traj_q)
    m = [[0 for i in range(len(query) + 1)] for j in range(len(cand) + 1)]  # 为方便后续计算，比字符串长度多了一列
    maxx = 0
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    subset = {}
    for i in range(len(cand)):
#         print("stage:", i)
        for j in range(len(query)):
            if abs(cand[i][0] - query[j][0]) < error_dict[0] and abs(cand[i][1] - query[j][1]) < error_dict[1]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i+1][j+1]> maxx:
                    maxx = m[i+1][j+1]
                if m[i + 1][j + 1] > minLen:
                    temp = distance(cand[i - m[i + 1][j + 1] + 1: i + 1], query[j - m[i + 1][j + 1] +1 : j + 1])
                    subset[(i - m[i + 1][j + 1] + 1 , i, j - m[i + 1][j + 1] + 1 , j)] = temp
                    if temp < subsim:
                        subsim = temp
                        subtraj = (i - m[i + 1][j + 1] + 1, i, j - m[i + 1][j + 1] + 1, j)
                        print(m[i + 1][j + 1] )
                        print('sub-trajectory', subtraj)
                        print('sub-similarity', subsim)
    return maxx, subsim, subtraj, subset
    