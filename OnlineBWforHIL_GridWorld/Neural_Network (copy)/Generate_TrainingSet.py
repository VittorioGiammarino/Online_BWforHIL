#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:46:32 2021

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

expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation_tabular(pi_hi_expert, pi_lo_expert, pi_b_expert)

max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = 50 #number of trajectories generated
TrainingSet_Array = []
Labels_Array = []
Reward_Array = []

for seed in range(30):
    [trajExpert, controlExpert, OptionsExpert, 
     TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch, nTraj, seed)
    ss = expert.Environment.stateSpace
    Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
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
    
    
    