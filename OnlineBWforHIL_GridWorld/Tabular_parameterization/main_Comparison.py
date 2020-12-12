#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:10:58 2020

@author: vittorio
"""


import World 
import BatchBW_HIL 
import OnlineBW_HIL
import numpy as np
import time

# %%

expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation(pi_hi_expert, pi_lo_expert, pi_b_expert)

max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = np.array([5])#, 10, 20, 40, 50, 100, 200]) #number of trajectories generated

Time_array_batch = np.empty((0))
Time_array_online = np.empty((0))
RewardExpert_array = np.empty((0))
STDExpert_array = np.empty((0))
RewardBatch_array = np.empty((0))
STDBatch_array = np.empty((0))
RewardOnline_array = np.empty((0))
STDOnline_array = np.empty((0))

for i in range(len(nTraj)):
    [trajExpert, controlExpert, OptionsExpert, 
    TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch, nTraj[i])
    ss = expert.Environment.stateSpace
    Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
    option_space = 2
    
    #Batch BW for HIL with tabular parameterization: Training
    Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space)
    N=10 #number of iterations for the BW algorithm
    start_batch_time = time.time()
    pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(N)
    end_batch_time = time.time()
    Batch_time = end_batch_time-start_batch_time
    Time_array_batch = np.append(Time_array_batch, Batch_time)

    # Online BW for HIL with tabular parameterization: Training
    Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space)
    T_min = nTraj[i]/2
    start_online_time = time.time()
    pi_hi_online, pi_lo_online, pi_b_online, chi, rho, phi  = Agent_OnlineHIL.Online_Baum_Welch(T_min)
    end_online_time = time.time()
    Online_time = end_online_time-start_online_time
    Time_array_online = np.append(Time_array_online, Online_time)
    
    # Expert
    AverageRewardExpert = np.sum(rewardExpert)/nTraj[i]
    STDExpert = np.std(rewardExpert)
    STDExpert_array = np.append(STDExpert_array, STDExpert)
    RewardExpert_array = np.append(RewardExpert_array, AverageRewardExpert)
    
    # Batch Agent Evaluation
    nTraj_eval = 100
    BatchSim = expert.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, 
     TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval)
    AverageRewardBatch = np.sum(rewardBatch)/nTraj_eval
    STDBatch = np.std(rewardBatch)
    RewardBatch_array = np.append(RewardBatch_array, AverageRewardBatch)
    STDBatch_array = np.append(STDBatch_array, STDBatch)

    # Online Agent Evaluation
    OnlineSim = expert.Simulation(pi_hi_online, pi_lo_online, pi_b_online)
    [trajOnline, controlOnline, OptionsOnline, 
     TerminationOnline, psiOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval)
    AverageRewardOnline = np.sum(rewardOnline)/nTraj_eval  
    STDOnline = np.std(rewardOnline)
    RewardOnline_array = np.append(RewardOnline_array, AverageRewardOnline)
    STDOnline_array = np.append(STDOnline_array, STDOnline)
    


# %% Comparison


    