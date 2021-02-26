#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:10:28 2021

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

with open('Models/Saved_Model_Expert/Q_trained.npy', 'rb') as f:
    Q_trained = np.load(f, allow_pickle=True)

n_bins = (3, 3, 6, 6)
Q_table = np.zeros(n_bins + (2,))
expert = World.CartPole.Expert(n_bins, 1, 1, Q_table)
lower_bounds = [expert.env.observation_space.low[0], -0.3, expert.env.observation_space.low[2], -np.radians(50) ]
upper_bounds = [expert.env.observation_space.high[0], 0.3, expert.env.observation_space.high[2], np.radians(50) ]
expert = World.CartPole.Expert(n_bins, lower_bounds, upper_bounds, Q_table)

max_epoch = 2000 #max iterations in the simulation per trajectory
nTraj = 3 #number of trajectories generated
TrainingSet_Array = []
Labels_Array = []
Reward_Array = []

for seed in range(10):
    TrainingSet, Labels, rewardExpert = expert.Evaluation(Q_trained, nTraj, max_epoch, seed)
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

