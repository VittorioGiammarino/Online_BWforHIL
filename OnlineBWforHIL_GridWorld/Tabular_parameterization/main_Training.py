#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:02:52 2020

@author: vittorio
"""
import World 
import BatchBW_HIL 
import OnlineBW_HIL
import numpy as np
import time

# %% Expert Policy Generation and simulation
expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = 3 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
  TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

# %% Batch BW for HIL with tabular parameterization: Training
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space)
N=10 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time

# %% Online BW for HIL with tabular parameterization: Training
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space)
T_min = 100

start_online_time = time.time()
pi_hi_online, pi_lo_online, pi_b_online, chi, rho, phi  = Agent_OnlineHIL.Online_Baum_Welch(T_min)
end_online_time = time.time()
Online_time = end_online_time-start_online_time

# start_online_time2 = time.time()
# pi_hi_online2, pi_lo_online2, pi_b_online2, phi2  = Agent_OnlineHIL.Online_Baum_Welch_together(T_min)
# end_online_time2 = time.time()
# Online_time2 = end_online_time2-start_online_time2

# %% Save Model

with open('Models/Saved_Model_Expert/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_expert)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_expert)

with open('Models/Saved_Model_Expert/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_expert)    
    
with open('Models/Saved_Model_Batch/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_batch)
    
with open('Models/Saved_Model_Batch/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_batch)

with open('Models/Saved_Model_Batch/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_batch)    
    
with open('Models/Saved_Model_Online/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_online)
    
with open('Models/Saved_Model_Online/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_online)

with open('Models/Saved_Model_Online/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_online)    
    
