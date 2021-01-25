#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:45:35 2021

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
    
with open('Comparison/Batch/List_TimeBatch.npy', 'rb') as f:
    List_TimeBatch = np.load(f)
    
# %%
expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation_tabular(pi_hi_expert, pi_lo_expert, pi_b_expert)

max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = np.array([1, 2, 5, 10, 20, 30, 50]) #number of trajectories generated

#%%
#seed
List_TimeOnline = []
List_RewardOnline = []
List_STDOnline = []
List_LikelihoodOnline = []
List_TimeLikelihoodOnline = []

#given seed, trajectories
Time_array_online = np.empty((0))
RewardOnline_array = np.empty((0))
STDOnline_array = np.empty((0))
Likelihood_online_list = []
time_likelihood_online_list =[]

for seed in range(1): #range(TrainingSet_Array.shape[0]):
    TrainingSet_tot = TrainingSet_Array[seed, :, :]
    Labels_tot = Labels_Array[seed, :, :]
    TimeBatch = List_TimeBatch[seed]
    for i in range(len(nTraj)):
        TrainingSet = np.concatenate((TrainingSet_tot[0:max_epoch*nTraj[i],:],TrainingSet_tot[0:max_epoch*nTraj[i],:],TrainingSet_tot[0:max_epoch*nTraj[i],:],TrainingSet_tot[0:max_epoch*nTraj[i],:]),axis=0)
        Labels = np.concatenate((Labels_tot[0:max_epoch*nTraj[i]],Labels_tot[0:max_epoch*nTraj[i]],Labels_tot[0:max_epoch*nTraj[i]],Labels_tot[0:max_epoch*nTraj[i]]),axis=0)
        option_space = 2
        
        #Stopping Time
        StoppingTime = TimeBatch[i]
    
        # Online BW for HIL with tabular parameterization: Training
        M_step_epoch = 30
        optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
        Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer)
        T_min = len(TrainingSet)/(4) - 20
        start_online_time = time.time()
        pi_hi_online, pi_lo_online, pi_b_online, likelihood_online, time_per_iteration = Agent_OnlineHIL.Online_Baum_Welch_together(T_min, StoppingTime)
        end_online_time = time.time()
        Online_time = end_online_time-start_online_time
        Time_array_online = np.append(Time_array_online, Online_time)  
        Likelihood_online_list.append(likelihood_online)
        time_likelihood_online_list.append(time_per_iteration)
    
        # Batch Agent Evaluation
        nTraj_eval = 100
        # Online Agent Evaluation
        OnlineSim = expert.Simulation_NN(pi_hi_online, pi_lo_online, pi_b_online)
        [trajOnline, controlOnline, OptionsOnline, 
         TerminationOnline, psiOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval, seed)
        AverageRewardOnline = np.sum(rewardOnline)/nTraj_eval  
        STDOnline = np.std(rewardOnline)
        RewardOnline_array = np.append(RewardOnline_array, AverageRewardOnline)
        STDOnline_array = np.append(STDOnline_array, STDOnline)
        
    List_TimeOnline.append(Time_array_online)
    List_RewardOnline.append(RewardOnline_array)
    List_STDOnline.append(STDOnline_array)
    List_LikelihoodOnline.append(Likelihood_online_list)
    List_TimeLikelihoodOnline.append(time_likelihood_online_list)


# %%

with open('Comparison/Online/List_TimeOnline.npy', 'wb') as f:
    np.save(f, List_TimeOnline)
    
with open('Comparison/Online/List_RewardOnline.npy', 'wb') as f:
    np.save(f, List_RewardOnline)
    
with open('Comparison/Online/List_STDOnline.npy', 'wb') as f:
    np.save(f, List_STDOnline)
    
with open('Comparison/Online/List_LikelihoodOnline.npy', 'wb') as f:
    np.save(f, List_LikelihoodOnline)
    
with open('Comparison/Online/List_TimeLikelihoodOnline.npy', 'wb') as f:
    np.save(f, List_TimeLikelihoodOnline)
