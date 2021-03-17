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
max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = 1 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,0)

# %% Hierarchical policy initialization Initialiazation 
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
seed=0
    
# %% Batch BW for HIL with tabular parameterization: Training
M_step_epoch = 10
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer) 
N=10 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood_batch, time_per_iteration = Agent_BatchHIL.Baum_Welch(N,1)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time
#evaluation
Batch_Plot = expert.Plot(pi_hi_batch, pi_lo_batch, pi_b_batch)
Batch_Plot.PlotHierachicalPolicy('Figures/FiguresBatch/Batch_High_policy_psi{}.eps','Figures/FiguresBatch/Batch_Action_option{}_psi{}.eps','Figures/FiguresBatch/Batch_Termination_option{}_psi{}.eps')
BatchSim = expert.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,seed)
best = np.argmax(rewardBatch)  
BatchSim.HILVideoSimulation(controlBatch[best][:], trajBatch[best][:], 
                            OptionsBatch[best][:], psiBatch[best][:],"Videos/VideosBatchAgent/sim_BatchBW.mp4")

# %% Online BW for HIL with tabular parameterization: Training
M_step_epoch = 30
Batch_time=200
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer) 
T_min = 50
start_online_time = time.time()
pi_hi_online, pi_lo_online, pi_b_online, likelihood_online, time_online = Agent_OnlineHIL.Online_Baum_Welch_together(T_min, Batch_time)
end_online_time = time.time()
Online_time = end_online_time-start_online_time
#evaluation
Online_Plot = expert.Plot(pi_hi_online, pi_lo_online, pi_b_online)
Online_Plot.PlotHierachicalPolicy('Figures/FiguresOnline/Online_High_policy_psi{}.eps','Figures/FiguresOnline/Online_Action_option{}_psi{}.eps','Figures/FiguresOnline/Online_Termination_option{}_psi{}.eps')
OnlineSim = expert.Simulation_NN(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, psiOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,seed)
best = np.argmax(rewardOnline)  
OnlineSim.HILVideoSimulation(controlOnline[best][:], trajOnline[best][:], 
                            OptionsOnline[best][:], psiOnline[best][:],"Videos/VideosOnlineAgent/sim_OnlineBW.mp4")

# %% Online BW modified for HIL with tabular parameterization: Training
M_step_epoch = 30
stopping_time = Batch_time
optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL_modified(TrainingSet, Labels, option_space, M_step_epoch, optimizer) 
T_min = 50
start_online_time = time.time()
pi_hi_online, pi_lo_online, pi_b_online, likelihood_online_mod, time_online = Agent_OnlineHIL.Online_Baum_Welch_together(T_min, Batch_time)
end_online_time = time.time()
Online_time = end_online_time-start_online_time
#evaluation
Online_Plot = expert.Plot(pi_hi_online, pi_lo_online, pi_b_online)
Online_Plot.PlotHierachicalPolicy('Figures/FiguresOnline/Online_High_policy_psi{}.eps','Figures/FiguresOnline/Online_Action_option{}_psi{}.eps','Figures/FiguresOnline/Online_Termination_option{}_psi{}.eps')
OnlineSim = expert.Simulation_NN(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, psiOnline, rewardOnline_mod] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,seed)
best = np.argmax(rewardOnline)  
OnlineSim.HILVideoSimulation(controlOnline[best][:], trajOnline[best][:], 
                            OptionsOnline[best][:], psiOnline[best][:],"Videos/VideosOnlineAgent/sim_OnlineBW.mp4")

# %% Likelihood comparison Expert
pi_lo_output_batch = []
for i in range(option_space):
    pi_lo_output_batch.append(pi_lo_batch[i](TrainingSet))

T = pi_lo_output_batch[0].shape[0]
action_space = pi_lo_output_batch[0].shape[1]
anomaly = []

for t in range(T):
    state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
    state_ID = expert.FindStateIndex(state)
    action = Labels[t]    
    partial = 0
    mu_temp = np.zeros((option_space))
    for o_past in range(option_space):
        for b in range(2):
            for o in range(option_space):
                if b == 0 and o == o_past:
                    pi_hi = 1
                elif b == 1:
                    pi_hi = pi_hi_expert[state_ID,o]
                else:
                    pi_hi = 0
                partial = partial + pi_lo_expert[state_ID, int(action), o]*pi_hi*pi_b_expert[state_ID,b,o_past]*pi_hi_expert[state_ID,o_past]
                    
    if partial == 0:
        anomaly.append(t)            
    if t == 0:
        likelihood_expert = partial
    else:
        likelihood_expert = (likelihood_expert + partial)

likelihood_expert = likelihood_expert/T


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
    
with open('Models/likelihood_batch.npy', 'wb') as f:
    np.save(f, likelihood_batch)

with open('Models/likelihood_online.npy', 'wb') as f:
    np.save(f, likelihood_online)