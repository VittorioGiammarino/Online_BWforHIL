#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:04:55 2021

@author: vittorio
"""

import World 
import BatchBW_HIL 
import OnlineBW_HIL
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
# %%

max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = 100 #number of trajectories generated
TrainingSet_Array = []
Labels_Array = []
Reward_Array = []

for seed in range(10):
    TrainingSet, Labels, rewardExpert = World.LunarLander.Expert.Evaluation(nTraj, max_epoch, seed)
    TrainingSet = np.round(TrainingSet,3)
    TrainingSet_Array.append(TrainingSet)
    Labels_Array.append(Labels)
    Reward_Array.append(rewardExpert)

# %%

with open('DataFromExpert/TrainingSet_Array.npy', 'wb') as f:
    np.save(f, TrainingSet_Array)
    
with open('DataFromExpert/Labels_Array.npy', 'wb') as f:
    np.save(f, Labels_Array)
    
with open('DataFromExpert/Reward_Array.npy', 'wb') as f:
    np.save(f, Reward_Array)
    