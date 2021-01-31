#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:49:08 2021

@author: vittorio
"""
import World 
import numpy as np

# %% Expert Policy Generation and simulation
with open('Models/Saved_Model_Expert/W_weights.npy', 'rb') as f:
    weights = np.load(f, allow_pickle=True)
    
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = 100 #number of trajectories generated
TrainingSet_Array = []
Labels_Array = []
Reward_Array = []

for seed in range(10):
    TrainingSet, Labels, Reward = World.Pendulum.Expert.Evaluation(weights, nTraj, max_epoch, seed)
    TrainingSet_Array.append(TrainingSet)
    Labels_Array.append(Labels)
    Reward_Array.append(Reward)

# %%

with open('DataFromExpert/TrainingSet_Array.npy', 'wb') as f:
    np.save(f, TrainingSet_Array)
    
with open('DataFromExpert/Labels_Array.npy', 'wb') as f:
    np.save(f, Labels_Array)
    
with open('DataFromExpert/Reward_Array.npy', 'wb') as f:
    np.save(f, Reward_Array)
    
    