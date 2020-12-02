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

# %% Expert Policy Generation and simulation
expert = World.TwoRewards.Expert()
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation_tabular(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = 10 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

# %% Batch BW for HIL with tabular parameterization: Training
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
M_step_epoch = 200
size_batch = 35
optimizer = keras.optimizers.Adamax(learning_rate=1e-4)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer)
N=15 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(N)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time
#evaluation
Batch_Plot = expert.Plot(pi_hi_batch, pi_lo_batch, pi_b_batch)
Batch_Plot.PlotHierachicalPolicy('Figures/FiguresBatch/Batch_High_policy_psi{}.eps','Figures/FiguresBatch/Batch_Action_option{}_psi{}.eps','Figures/FiguresBatch/Batch_Termination_option{}_psi{}.eps')
BatchSim = expert.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardBatch)  
BatchSim.HILVideoSimulation(controlBatch[best][:], trajBatch[best][:], 
                            OptionsBatch[best][:], psiBatch[best][:],"Videos/VideosBatchAgent/sim_BatchBW.mp4")

# %% Online BW for HIL with tabular parameterization: Training
M_step_epoch = 1
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer)
T_min = 1000
start_online_time = time.time()
pi_hi_online, pi_lo_online, pi_b_online = Agent_OnlineHIL.Online_Baum_Welch(T_min)
end_online_time = time.time()
Online_time = end_online_time-start_online_time
#evaluation
Online_Plot = expert.Plot(pi_hi_online, pi_lo_online, pi_b_online)
Online_Plot.PlotHierachicalPolicy('Figures/FiguresOnline/Online_High_policy_psi{}.eps','Figures/FiguresOnline/Online_Action_option{}_psi{}.eps','Figures/FiguresOnline/Online_Termination_option{}_psi{}.eps')
OnlineSim = expert.Simulation_NN(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, psiOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardOnline)  
OnlineSim.HILVideoSimulation(controlOnline[best][:], trajOnline[best][:], 
                            OptionsOnline[best][:], psiOnline[best][:],"Videos/VideosOnlineAgent/sim_OnlineBW.mp4")



# %% Save Model

with open('Models/Saved_Model_Expert/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_expert)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_expert)

with open('Models/Saved_Model_Expert/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_expert)   
    
BatchBW_HIL.NN_PI_HI.save(pi_hi_batch, 'Models/Saved_Model_Batch/pi_hi_NN')
for i in range(option_space):
    BatchBW_HIL.NN_PI_LO.save(pi_lo_batch[i], 'Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i))
    BatchBW_HIL.NN_PI_B.save(pi_b_batch[i], 'Models/Saved_Model_Batch/pi_b_NN_{}'.format(i))

    
OnlineBW_HIL.NN_PI_HI.save(pi_hi_online, 'Models/Saved_Model_Online/pi_hi_NN')
for i in range(option_space):
    OnlineBW_HIL.NN_PI_LO.save(pi_lo_online[i], 'Models/Saved_Model_Online/pi_lo_NN_{}'.format(i))
    OnlineBW_HIL.NN_PI_B.save(pi_b_online[i], 'Models/Saved_Model_Online/pi_b_NN_{}'.format(i))
