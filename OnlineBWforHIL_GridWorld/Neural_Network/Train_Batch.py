#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:09:50 2021

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

# %% Expert Data

with open('DataFromExpert/TrainingSet_Array.npy', 'rb') as f:
    TrainingSet_Array = np.load(f)
    
with open('DataFromExpert/Labels_Array.npy', 'rb') as f:
    Labels_Array = np.load(f)
    
# %%
expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation_tabular(pi_hi_expert, pi_lo_expert, pi_b_expert)

max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = np.array([1, 2, 5, 10, 20, 30, 50]) #number of trajectories generated

#%%
#seed
List_TimeBatch = []
List_RewardBatch = []
List_STDBatch = []
List_LikelihoodBatch = []
List_TimeLikelihoodBatch = []

#given seed, trajectories
Time_array_batch = np.empty((0))
RewardBatch_array = np.empty((0))
STDBatch_array = np.empty((0))
Likelihood_batch_list = []
time_likelihood_batch_list =[]

for seed in range(1): #range(TrainingSet_Array.shape[0]):
    TrainingSet_tot = TrainingSet_Array[seed, :, :]
    Labels_tot = Labels_Array[seed, :, :]
    for i in range(len(nTraj)):
        TrainingSet = TrainingSet_tot[0:max_epoch*nTraj[i],:]
        Labels = Labels_tot[0:max_epoch*nTraj[i]]
        option_space = 2
    
        #Batch BW for HIL with tabular parameterization: Training
        M_step_epoch = 50
        size_batch = 32
        if np.mod(len(TrainingSet),size_batch)==0:
            size_batch = size_batch + 1
        optimizer = keras.optimizers.Adamax(learning_rate=1e-2)    
        Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer)
        N=20 #number of iterations for the BW algorithm
        start_batch_time = time.time()
        pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood_batch, time_per_iteration = Agent_BatchHIL.Baum_Welch(N, 1)
        end_batch_time = time.time()
        Batch_time = end_batch_time-start_batch_time
        Time_array_batch = np.append(Time_array_batch, Batch_time)
        Likelihood_batch_list.append(likelihood_batch)
        time_likelihood_batch_list.append(time_per_iteration)
    
        # Batch Agent Evaluation
        nTraj_eval = 100
        BatchSim = expert.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
        [trajBatch, controlBatch, OptionsBatch, 
         TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval, seed)
        AverageRewardBatch = np.sum(rewardBatch)/nTraj_eval
        STDBatch = np.std(rewardBatch)
        RewardBatch_array = np.append(RewardBatch_array, AverageRewardBatch)
        STDBatch_array = np.append(STDBatch_array, STDBatch)
        
    List_TimeBatch.append(Time_array_batch)
    List_RewardBatch.append(RewardBatch_array)
    List_STDBatch.append(STDBatch_array)
    List_LikelihoodBatch.append(Likelihood_batch_list)
    List_TimeLikelihoodBatch.append(time_likelihood_batch_list)
    
    
    
# %%

with open('Comparison/Batch/List_TimeBatch.npy', 'wb') as f:
    np.save(f, List_TimeBatch)
    
with open('Comparison/Batch/List_RewardBatch.npy', 'wb') as f:
    np.save(f, List_RewardBatch)
    
with open('Comparison/Batch/List_STDBatch.npy', 'wb') as f:
    np.save(f, List_STDBatch)
    
with open('Comparison/Batch/List_LikelihoodBatch.npy', 'wb') as f:
    np.save(f, List_LikelihoodBatch)
    
with open('Comparison/Batch/List_TimeLikelihoodBatch.npy', 'wb') as f:
    np.save(f, List_TimeLikelihoodBatch)
        



