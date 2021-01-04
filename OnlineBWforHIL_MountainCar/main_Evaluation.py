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
bc_data_dir = 'Expert/Data'
TrainingSet, labels = BatchBW_HIL.PreprocessData(bc_data_dir)

TrainingSet = np.round(TrainingSet[:,:],3)
Labels = labels[:]

Steps2goal_Expert = World.MountainCar.Expert.AverageExpert(TrainingSet)

# %% Batch Agent
max_epoch = 1000
nTraj = 300
BatchSim = World.MountainCar.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, flagBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

averageBatch = np.empty((0))
success_percentageBatch = np.empty((0))
length_trajBatch = np.empty((0))
for j in range(len(trajBatch)):
    length_trajBatch = np.append(length_trajBatch, len(trajBatch[j][:]))
averageBatch = np.append(averageBatch,np.divide(np.sum(length_trajBatch),len(length_trajBatch)))
success_percentageBatch = np.append(success_percentageBatch,np.divide(np.sum(flagBatch),len(length_trajBatch)))


# %% Online Agent
OnlineSim = World.MountainCar.Simulation(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, flagOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

averageOnline = np.empty((0))
success_percentageOnline = np.empty((0))
length_trajOnline = np.empty((0))
for j in range(len(trajOnline)):
    length_trajOnline = np.append(length_trajOnline, len(trajOnline[j][:]))
averageOnline = np.append(averageOnline,np.divide(np.sum(length_trajOnline),len(length_trajOnline)))
success_percentageOnline = np.append(success_percentageOnline,np.divide(np.sum(flagOnline),len(length_trajOnline)))


# %% Plot Likelihood 


x_likelihood_batch = np.linspace(0, len(likelihood_batch)-1, len(likelihood_batch)) 
x_likelihood_online = np.linspace(0,len(likelihood_online)-1,len(likelihood_online))

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

   