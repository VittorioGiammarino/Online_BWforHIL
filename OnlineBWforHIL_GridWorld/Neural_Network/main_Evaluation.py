#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:59 2020

@author: vittorio
"""
import World
import numpy as np
import OnlineBW_HIL
import BatchBW_HIL 
import matplotlib.pyplot as plt

with open('Models/likelihood_batch.npy', 'rb') as f:
    likelihood_batch = np.load(f, allow_pickle=True)

with open('Models/likelihood_online.npy', 'rb') as f:
    likelihood_online = np.load(f, allow_pickle=True)

with open('Models/Saved_Model_Expert/pi_hi.npy', 'rb') as f:
    pi_hi_expert = np.load(f)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'rb') as f:
    pi_lo_expert = np.load(f)

with open('Models/Saved_Model_Expert/pi_b.npy', 'rb') as f:
    pi_b_expert = np.load(f)   
        
pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN')
pi_lo_batch = []
pi_b_batch = []
option_space = 2
for i in range(option_space):
    pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i)))
    pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}'.format(i)))

pi_hi_online = OnlineBW_HIL.NN_PI_HI.load('Models/Saved_Model_Online/pi_hi_NN')     
pi_lo_online = []
pi_b_online = []
for i in range(option_space):
    pi_lo_online.append(OnlineBW_HIL.NN_PI_LO.load('Models/Saved_Model_Online/pi_lo_NN_{}'.format(i)))
    pi_b_online.append(OnlineBW_HIL.NN_PI_B.load('Models/Saved_Model_Online/pi_b_NN_{}'.format(i)))
        


# %% Expert
expert = World.TwoRewards.Expert()
ExpertSim = expert.Simulation_tabular(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = 100 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardExpert)   
ExpertSim.HILVideoSimulation(controlExpert[best][:], trajExpert[best][:], 
                             OptionsExpert[best][:], psiExpert[best][:],"Videos/VideosExpert/sim_HierarchExpert.mp4")

# %% Batch Agent
Batch_Plot = expert.Plot(pi_hi_batch, pi_lo_batch, pi_b_batch)
Batch_Plot.PlotHierachicalPolicy('Figures/FiguresBatch/Batch_High_policy_psi{}.eps','Figures/FiguresBatch/Batch_Action_option{}_psi{}.eps','Figures/FiguresBatch/Batch_Termination_option{}_psi{}.eps')
BatchSim = expert.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardBatch)  
BatchSim.HILVideoSimulation(controlBatch[best][:], trajBatch[best][:], 
                            OptionsBatch[best][:], psiBatch[best][:],"Videos/VideosBatchAgent/sim_BatchBW.mp4")

# %% Online Agent
Online_Plot = expert.Plot(pi_hi_online, pi_lo_online, pi_b_online)
Online_Plot.PlotHierachicalPolicy('Figures/FiguresOnline/Online_High_policy_psi{}.eps','Figures/FiguresOnline/Online_Action_option{}_psi{}.eps','Figures/FiguresOnline/Online_Termination_option{}_psi{}.eps')
OnlineSim = expert.Simulation_NN(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, psiOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardOnline)  
OnlineSim.HILVideoSimulation(controlOnline[best][:], trajOnline[best][:], 
                             OptionsOnline[best][:], psiOnline[best][:],"Videos/VideosOnlineAgent/sim_OnlineBW.mp4")

# %% Comparison
AverageRewardExpert = np.sum(rewardExpert)/nTraj
AverageRewardBatch = np.sum(rewardBatch)/nTraj
AverageRewardOnline = np.sum(rewardOnline)/nTraj

# %% Plot Likelihood 


x_likelihood_batch = np.linspace(1, len(likelihood_batch), len(likelihood_batch)) 
x_likelihood_online = np.linspace(1,len(likelihood_online),len(likelihood_online))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Batch iterations' , color=color)
ax1.set_ylabel('likelihood')
ax1.plot(x_likelihood_batch, likelihood_batch, '-d', color=color)
ax1.tick_params(axis='x', labelcolor=color)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_xlabel('Online iterations' , color=color)  # we already handled the x-label with ax1
ax2.plot(x_likelihood_online, likelihood_online, color=color)
ax2.tick_params(axis='x', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Figures/likelihood.eps', format='eps')
plt.show()

   