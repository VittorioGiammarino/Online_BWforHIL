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
from tensorflow import keras
import time
import matplotlib.pyplot as plt
  


# %% Expert Policy Generation and simulation
with open('Models/Saved_Model_Expert/Q_trained.npy', 'rb') as f:
    Q_trained = np.load(f, allow_pickle=True)

# %%
n_bins = (8, 8, 8, 8, 4, 4)
Q_table = np.zeros(n_bins + (2,))
expert = World.Acrobot.Expert.Expert_Q_learning(n_bins, Q_table)
max_epoch = 20000
nTraj = 5
TrainingSet, Labels, Reward = expert.Evaluation(Q_trained, nTraj, max_epoch)
TrainingSet = np.round(TrainingSet[0:700,:],3)
Labels = Labels[0:700]

# %% Hierarchical policy initialization 
option_space = 2
    
# %% Batch BW for HIL with tabular parameterization: Training
M_step_epoch = 50
size_batch = 33
optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer) 
N=10 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood_batch = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time
#evaluation
max_epoch = 20000
nTraj = 3
BatchSim = World.Acrobot.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch, Labels)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, RewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
x, u, o, b = BatchSim.HILVideoSimulation('Videos/VideosBatch/Simulation', max_epoch)
World.Acrobot.Plot(x, u, o, b, 'Figures/FiguresBatch/Batch_simulation.eps')

# %% Online BW for HIL with tabular parameterization: Training
M_step_epoch = 1
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer) 
T_min = 400
start_online_time = time.time()
pi_hi_online, pi_lo_online, pi_b_online, likelihood_online = Agent_OnlineHIL.Online_Baum_Welch_together(T_min)
end_online_time = time.time()
Online_time = end_online_time-start_online_time
#evaluation
OnlineSim = World.Acrobot.Simulation(pi_hi_online, pi_lo_online, pi_b_online, Labels)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, RewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
x, u, o, b = OnlineSim.HILVideoSimulation('Videos/VideosOnline/Simulation', max_epoch)
World.Acrobot.Plot(x, u, o, b, 'Figures/FiguresOnline/Online_simulation.eps')


# %% Save Model
 
    
BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}'.format(i))

    
OnlineBW_HIL.NN_PI_HI.save(pi_hi_online, 'Models/Saved_Model_Online/pi_hi_NN')
for i in range(option_space):
    OnlineBW_HIL.NN_PI_LO.save(pi_lo_online[i], 'Models/Saved_Model_Online/pi_lo_NN_{}'.format(i))
    OnlineBW_HIL.NN_PI_B.save(pi_b_online[i], 'Models/Saved_Model_Online/pi_b_NN_{}'.format(i))
    
with open('Models/likelihood_batch.npy', 'wb') as f:
    np.save(f, likelihood_batch)

with open('Models/likelihood_online.npy', 'wb') as f:
    np.save(f, likelihood_online)
    