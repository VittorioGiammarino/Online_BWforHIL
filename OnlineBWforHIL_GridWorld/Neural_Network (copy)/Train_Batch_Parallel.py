#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:08:04 2021

@author: vittorio
"""

import World 
import BatchBW_HIL 
import OnlineBW_HIL
import numpy as np
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
from tensorflow import keras
import multiprocessing
import multiprocessing.pool

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
nTraj = np.array([1, 2, 5, 10, 20])#, 30, 50]) #number of trajectories generated

# %%

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def DifferentTrainingSet(i, nTraj, TrainingSet_tot, Labels_tot, seed):
    max_epoch = 100    
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
    # Time_array_batch = np.append(Time_array_batch, Batch_time)
    # Likelihood_batch_list.append(likelihood_batch)
    # time_likelihood_batch_list.append(time_per_iteration)
    
    # Batch Agent Evaluation
    nTraj_eval = 100
    BatchSim = expert.Simulation_NN(pi_hi_batch, pi_lo_batch, pi_b_batch)
    [trajBatch, controlBatch, OptionsBatch, 
    TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj_eval, seed)
    AverageRewardBatch = np.sum(rewardBatch)/nTraj_eval
    STDBatch = np.std(rewardBatch)
    # RewardBatch_array = np.append(RewardBatch_array, AverageRewardBatch)
    # STDBatch_array = np.append(STDBatch_array, STDBatch)
    
    
    return Batch_time, likelihood_batch, time_per_iteration, AverageRewardBatch, STDBatch


def train(seed, TrainingSet_Array, Labels_Array, max_epoch, nTraj):
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

    TrainingSet_tot = TrainingSet_Array[seed, :, :]
    Labels_tot = Labels_Array[seed, :, :]
        
    pool = multiprocessing.Pool(processes=1)
    args = [(i, nTraj, TrainingSet_tot, Labels_tot, seed) for i in range(len(nTraj))]
    givenSeed_training_results = pool.starmap(DifferentTrainingSet, args) 
    
    pool.close()
    pool.join()
    
    for i in range(len(nTraj)):
        Time_array_batch = np.append(Time_array_batch, givenSeed_training_results[i][0]) 
        Likelihood_batch_list.append(givenSeed_training_results[i][1])
        time_likelihood_batch_list.append(givenSeed_training_results[i][2])
        RewardBatch_array = np.append(RewardBatch_array, givenSeed_training_results[i][3])
        STDBatch_array = np.append(STDBatch_array, givenSeed_training_results[i][4])
        
    List_TimeBatch.append(Time_array_batch)
    List_RewardBatch.append(RewardBatch_array)
    List_STDBatch.append(STDBatch_array)
    List_LikelihoodBatch.append(Likelihood_batch_list)
    List_TimeLikelihoodBatch.append(time_likelihood_batch_list)
        
    return List_TimeBatch, List_RewardBatch, List_STDBatch, List_LikelihoodBatch, List_TimeLikelihoodBatch


pool = MyPool(30)
args = [(seed, TrainingSet_Array, Labels_Array, max_epoch, nTraj) for seed in range(30)]
results_batch = pool.starmap(train, args) 
pool.close()
pool.join()

# %%

with open('Comparison/Batch/results_batch.npy', 'wb') as f:
    np.save(f, results_batch)
