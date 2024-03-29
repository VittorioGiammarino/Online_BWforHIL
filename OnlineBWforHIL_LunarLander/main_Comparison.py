#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:36:12 2020

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
with open('DataFromExpert/Reward_Array.npy', 'rb') as f:
    Reward_Array = np.load(f)

with open('Comparison/Batch/results_batch.npy', 'rb') as f:
    results_batch = np.load(f, allow_pickle=True).tolist()
    
with open('Comparison/Online/results_online.npy', 'rb') as f:
    results_online = np.load(f, allow_pickle=True).tolist()
    
# %% pre-processing batch

List_TimeBatch = []
List_RewardBatch = []
List_STDBatch = []
List_LikelihoodBatch = []
List_TimeLikelihoodBatch = []

for seed in range(len(results_batch)):
    if seed == 0:
        List_TimeBatch.append(results_batch[seed][0][0])
        List_RewardBatch.append(results_batch[seed][1][0])
        List_STDBatch.append(results_batch[seed][2][0])

    else:
        List_TimeBatch[0] = np.add(List_TimeBatch[0],results_batch[seed][0][0])
        List_RewardBatch[0] = np.add(List_RewardBatch[0], results_batch[seed][1][0])
        List_STDBatch[0] = np.add(List_STDBatch[0], results_batch[seed][2][0])

    List_LikelihoodBatch.append(results_batch[seed][3][0])
    List_TimeLikelihoodBatch.append(results_batch[seed][4][0])
       
# normalize 
List_TimeBatch[0] = np.divide(List_TimeBatch[0],len(results_batch))
List_RewardBatch[0] = np.divide(List_RewardBatch[0], len(results_batch))
List_STDBatch[0] = np.divide(List_STDBatch[0], len(results_batch))

List_RewardBatch = np.zeros((len(results_batch), len(results_batch[seed][1][0])))

for trajs in range(len(results_batch[seed][1][0])):
    for seed in range(len(results_batch)):
        List_RewardBatch[seed, trajs] = results_batch[seed][1][0][trajs]

List_STDBatch = np.std(List_RewardBatch, axis=0)
List_RewardBatch = np.mean(List_RewardBatch, axis=0)


# %% pre-processing online

List_RewardOnline = np.zeros((len(results_online)))
List_STDOnline = np.zeros((len(results_online)))

for trajs in range(len(results_online)):
    Nseed = len(results_online[trajs])
    temp_reward = np.zeros((Nseed))
    for seed in range(Nseed):
        temp_reward[seed] = results_online[trajs][seed][1][0][0]
        
    List_RewardOnline[trajs] = np.mean(temp_reward)
    List_STDOnline[trajs] = np.std(temp_reward)
    

# %% Plot
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = np.array([1, 2, 5, 10]) #number of trajectories generated
Samples = max_epoch*nTraj

Reward_Expert = np.sum(Reward_Array)/(Reward_Array.shape[0]*Reward_Array.shape[1])
STDExpert = np.sum(np.std(Reward_Array, axis = 1))/len(np.std(Reward_Array, axis = 1))

size = len(List_RewardOnline)

fig, ax = plt.subplots()
plt.xscale('log')
plt.xticks(Samples, labels=['200', '400', '1000', '2k', '3k'])
clrs = sns.color_palette("husl", 5)
ax.plot(Samples[0:size], List_RewardOnline, label='Online-BW', c=clrs[0])
ax.fill_between(Samples[0:size], List_RewardOnline-List_STDOnline, List_RewardOnline+List_STDOnline, alpha=0.1, facecolor=clrs[0])
ax.plot(Samples[0:size], List_RewardBatch[0:size], label = 'Batch-BW', c=clrs[1])
ax.fill_between(Samples[0:size], List_RewardBatch[0:size]-List_STDBatch[0:size], List_RewardBatch[0:size]+List_STDBatch[0:size], alpha=0.1, facecolor=clrs[1])
ax.plot(Samples[0:size], Reward_Expert*np.ones(size), label='Expert', c=clrs[2])
# ax.fill_between(Samples[0:size], Reward_Expert*np.ones(size)-STDExpert, Reward_Expert+STDExpert, alpha=0.1, facecolor=clrs[2])
ax.legend(loc=4, facecolor = '#d8dcd6')
ax.set_xlabel('Training Samples')
ax.set_ylabel('Average Reward')
ax.set_title('Lunar Lander')
plt.savefig('Figures/Comparison/Reward_LL.png', format='png')

# fig_time, ax_time = plt.subplots()
# plt.xscale('log')
# plt.xticks(Samples, labels=['100', '200', '500', '1k', '2k'])
# ax_time.plot(Samples, List_TimeOnline[0]/3600, label='Online-BW', c=clrs[0])
# ax_time.plot(Samples, List_TimeBatch[0]/3600,  label = 'Batch-BW', c=clrs[1])
# ax_time.legend(loc=0, facecolor = '#d8dcd6')
# ax_time.set_xlabel('Training Samples')
# ax_time.set_ylabel('Running Time [h]')
# ax_time.set_title('Grid World')
# plt.savefig('Figures/Comparison/Time_LL.eps', format='eps')   

seed = 0
trial = 3
fig, ax1 = plt.subplots()
ax1.set_xlabel('time [h]' , color='k')
ax1.set_ylabel('likelihood')
ax1.plot(np.divide(List_TimeLikelihoodBatch[seed][trial],3600), List_LikelihoodBatch[seed][trial], '-d', color='tab:red', label='Batch-BW')
ax1.plot(np.divide(results_online[trial][seed][4][0][0],3600), results_online[trial][seed][3][0][0], '-', color='tab:blue', label='Online-BW')
ax1.tick_params(axis='x', labelcolor='k')
ax1.legend(loc=0, facecolor = '#d8dcd6')
ax1.set_title('{} Training Samples'.format(Samples[trial]))
plt.savefig('Figures/likelihood_comparison_Samples{}_Seed{}.eps'.format(Samples[trial], seed), format='eps')
# %% reward Scaled

Reward_Expert = np.sum(Reward_Array)/(Reward_Array.shape[0]*Reward_Array.shape[1])
STDExpert = np.sum(np.std(Reward_Array, axis = 1))/len(np.std(Reward_Array, axis = 1))

size = len(nTraj)

# fig, ax = plt.subplots()
# plt.xscale('log')
# plt.xticks(Samples, labels=['100', '200', '500', '1k', '2k'])
# clrs = sns.color_palette("husl", 5)
# ax.plot(Samples, List_RewardOnline[0], label='Online-BW', c=clrs[0])
# ax.fill_between(Samples, List_RewardOnline[0]-List_STDOnline[0], List_RewardOnline[0]+List_STDOnline[0], alpha=0.1, facecolor=clrs[0])
# ax.plot(Samples, List_RewardBatch[0], label = 'Batch-BW', c=clrs[1])
# ax.fill_between(Samples, List_RewardBatch[0]-List_STDBatch[0], List_RewardBatch[0]+List_STDBatch[0], alpha=0.1, facecolor=clrs[1])
# ax.plot(Samples, Reward_Expert*np.ones(len(nTraj)), label='Expert', c=clrs[2])
# ax.fill_between(Samples, Reward_Expert*np.ones(len(nTraj))-STDExpert, Reward_Expert+STDExpert, alpha=0.1, facecolor=clrs[2])
# ax.legend(loc=4, facecolor = '#d8dcd6')
# ax.set_xlabel('Training Samples')
# ax.set_ylabel('Average Reward')
# ax.set_title('Grid World')
# plt.savefig('Figures/Comparison/Reward_GridWorld_NN.png', format='png')

fig, ax = plt.subplots()
plt.xscale('log')
plt.xticks(Samples, labels=['100', '200', '500', '1k', '2k'])
clrs = sns.color_palette("husl", 5)
ax.plot(Samples, List_RewardOnline[0:size]/Reward_Expert, '-d', label='Online-BW', c=clrs[0])
ax.fill_between(Samples, List_RewardOnline[0:size]/Reward_Expert-List_STDOnline[0:size]/Reward_Expert, 
                List_RewardOnline[0:size]/Reward_Expert+List_STDOnline[0:size]/Reward_Expert, alpha=0.1, facecolor=clrs[0])
ax.plot(Samples, List_RewardBatch[0:size]/Reward_Expert, '-d', label = 'Batch-BW', c=clrs[1])
ax.fill_between(Samples, List_RewardBatch[0:size]/Reward_Expert-List_STDBatch[0:size]/Reward_Expert, 
                List_RewardBatch[0:size]/Reward_Expert+List_STDBatch[0:size]/Reward_Expert, alpha=0.1, facecolor=clrs[1])
ax.plot(Samples, Reward_Expert*np.ones(len(nTraj))/Reward_Expert, label='Expert', c=clrs[2])
#ax.fill_between(Samples, Reward_Expert*np.ones(len(nTraj))-STDExpert, Reward_Expert+STDExpert, alpha=0.1, facecolor=clrs[2])
ax.legend(loc=4, facecolor = '#d8dcd6')
ax.set_xlabel('Training Samples')
ax.set_ylabel('Average Reward')
ax.set_title('Lunar Lander')
plt.savefig('Figures/Comparison/Reward_LL_3_scaled.png', format='png')

# %%

max_epoch = 1000 #max iterations in the simulation per trajectory
nTraj = np.array([1, 2, 3, 5, 10, 15, 20]) #number of trajectories generated

#%%
Time_array_batch = np.empty((0))
Time_array_online = np.empty((0))
RewardExpert_array = np.empty((0))
STDExpert_array = np.empty((0))
RewardBatch_array = np.empty((0))
STDBatch_array = np.empty((0))
RewardOnline_array = np.empty((0))
STDOnline_array = np.empty((0))
Likelihood_online_list = []
Likelihood_batch_list = []

seed = 0

for i in range(len(nTraj)):
    TrainingSet, Labels, rewardExpert = World.LunarLander.Expert.Evaluation(nTraj[i], max_epoch, seed)
    TrainingSet = np.round(TrainingSet,3)
    option_space = 2
    
    # Online BW for HIL with tabular parameterization: Training
    M_step_epoch = 5
    optimizer = keras.optimizers.Adamax(learning_rate=1e-2)
    Agent_OnlineHIL = OnlineBW_HIL.OnlineHIL(TrainingSet, Labels, option_space, M_step_epoch, optimizer)
    T_min = len(TrainingSet)-100
    start_online_time = time.time()
    pi_hi_online, pi_lo_online, pi_b_online, likelihood_online = Agent_OnlineHIL.Online_Baum_Welch_together(T_min)
    end_online_time = time.time()
    Online_time = end_online_time-start_online_time
    Time_array_online = np.append(Time_array_online, Online_time)  
    Likelihood_online_list.append(likelihood_online)
    
    #Batch BW for HIL with tabular parameterization: Training
    M_step_epoch = 20
    size_batch = 32
    if np.mod(len(TrainingSet),size_batch)==0:
        size_batch = size_batch + 1
    optimizer = keras.optimizers.Adamax(learning_rate=1e-3)    
    Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer)
    N=15 #number of iterations for the BW algorithm
    start_batch_time = time.time()
    pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood_batch = Agent_BatchHIL.Baum_Welch(N, likelihood_online[-1])
    end_batch_time = time.time()
    Batch_time = end_batch_time-start_batch_time
    Time_array_batch = np.append(Time_array_batch, Batch_time)
    Likelihood_batch_list.append(likelihood_batch)
    
    # Expert
    AverageRewardExpert = np.sum(rewardExpert)/nTraj[i]
    STDExpert = np.std(rewardExpert)
    STDExpert_array = np.append(STDExpert_array, STDExpert)
    RewardExpert_array = np.append(RewardExpert_array, AverageRewardExpert)
    
    # Batch Agent Evaluation
    nTraj_eval = 100
    BatchSim = World.LunarLander.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch, Labels)
    [trajBatch, controlBatch, OptionsBatch, 
     TerminationBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval, seed)
    AverageRewardBatch = np.sum(rewardBatch)/nTraj_eval
    STDBatch = np.std(rewardBatch)
    RewardBatch_array = np.append(RewardBatch_array, AverageRewardBatch)
    STDBatch_array = np.append(STDBatch_array, STDBatch)

    # Online Agent Evaluation
    OnlineSim = World.LunarLander.Simulation(pi_hi_online, pi_lo_online, pi_b_online, Labels)
    [trajOnline, controlOnline, OptionsOnline, 
     TerminationOnline, rewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval, seed)
    AverageRewardOnline = np.sum(rewardOnline)/nTraj_eval  
    STDOnline = np.std(rewardOnline)
    RewardOnline_array = np.append(RewardOnline_array, AverageRewardOnline)
    STDOnline_array = np.append(STDOnline_array, STDOnline)

# %%
Training_samples = max_epoch*np.array([1, 2, 3, 5])
    
Array_results = np.concatenate((RewardOnline_array.reshape(1,len(Training_samples)), RewardBatch_array.reshape(1,len(Training_samples)), 
                                RewardExpert_array.reshape(1,len(Training_samples)), STDOnline_array.reshape(1,len(Training_samples)), 
                                STDBatch_array.reshape(1,len(Training_samples)), STDExpert_array.reshape(1,len(Training_samples)),
                                Time_array_online.reshape(1,len(Training_samples)), Time_array_batch.reshape(1,len(Training_samples))), axis=0)

with open('Comparison/Array_results.npy', 'wb') as f:
    np.save(f, Array_results)
    
# %% plot Comparison
Training_samples = max_epoch*np.array([1, 2, 3, 5])

RewardOnline_array = Array_results[0,:]
RewardBatch_array = Array_results[1,:]
RewardExpert_array = Array_results[2,:]
STDOnline_array = Array_results[3,:]
STDBatch_array = Array_results[4,:]
STDExpert_array = Array_results[5,:]
Time_array_online = Array_results[6,:]
Time_array_batch = Array_results[7,:]

fig, ax = plt.subplots()
plt.xscale('log')
plt.xticks(Training_samples, labels=['1', '2', '3', '5'])
clrs = sns.color_palette("husl", 5)
ax.plot(Training_samples, RewardOnline_array, label='Online-BW', c=clrs[0])
ax.fill_between(Training_samples, RewardOnline_array-STDOnline_array, RewardOnline_array+STDOnline_array ,alpha=0.1, facecolor=clrs[0])
ax.plot(Training_samples, RewardBatch_array, label = 'Batch-BW', c=clrs[1])
ax.fill_between(Training_samples, RewardBatch_array-STDBatch_array, RewardBatch_array+STDBatch_array ,alpha=0.1, facecolor=clrs[1])
ax.plot(Training_samples, RewardExpert_array, label='Expert', c=clrs[2])
ax.fill_between(Training_samples, RewardExpert_array-STDExpert_array, RewardExpert_array+STDExpert_array ,alpha=0.1, facecolor=clrs[2])
ax.legend(loc=4, facecolor = '#d8dcd6')
ax.set_xlabel('Trajectories')
ax.set_ylabel('Average Reward')
ax.set_title('Lunar Lander')
plt.savefig('Figures/Comparison/Reward_LunarLander.png', format='png')


fig_time, ax_time = plt.subplots()
plt.xscale('log')
plt.xticks(Training_samples, labels=['1', '2', '3', '5'])
ax_time.plot(Training_samples, Time_array_online/3600, label='Online-BW', c=clrs[0])
ax_time.plot(Training_samples, Time_array_batch/3600,  label = 'Batch-BW', c=clrs[1])
ax_time.legend(loc=0, facecolor = '#d8dcd6')
ax_time.set_xlabel('Trajectories')
ax_time.set_ylabel('Running Time [h]')
ax_time.set_title('Lunar Lander')
plt.savefig('Figures/Comparison/Time_LunarLander.eps', format='eps')   

# %% Plot Likelihood 

trial = 4

x_likelihood_batch = np.linspace(0, len(Likelihood_batch_list[trial])-1, len(Likelihood_batch_list[trial])) 
x_likelihood_online = np.linspace(0,len(Likelihood_online_list[trial])-1,len(Likelihood_online_list[trial]))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Batch iterations' , color=color)
ax1.set_ylabel('likelihood')
ax1.plot(x_likelihood_batch, Likelihood_batch_list[trial], '-d', color=color)
ax1.tick_params(axis='x', labelcolor=color)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_xlabel('Online iterations' , color=color)  # we already handled the x-label with ax1
ax2.plot(x_likelihood_online, Likelihood_online_list[trial], color=color)
ax2.tick_params(axis='x', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Figures/likelihood_trial{}.eps'.format(trial), format='eps')
plt.show()

