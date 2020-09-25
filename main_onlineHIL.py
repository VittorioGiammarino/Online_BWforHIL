#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:14:27 2020

@author: vittorio
"""

import numpy as np
import Simulation as sim
import matplotlib.pyplot as plt
import HierarchicalImitationLearning as hil

# %% Expert
theta_hi_1 = 0.05
theta_hi_2 = 0.05
pi_hi = np.array([[theta_hi_1, 1-theta_hi_1], [1-theta_hi_2, theta_hi_2]])
option_space = 2

theta_lo_1 = 0.8
theta_lo_2 = 0.9
theta_lo_3 = 0.95
theta_lo_4 = 0.8
pi_lo = np.array([[theta_lo_1, 1-theta_lo_1], [1-theta_lo_2, theta_lo_2], [theta_lo_3, 1-theta_lo_3], [1-theta_lo_4, theta_lo_4]])
pi_lo = pi_lo.reshape((2,2,2))
action_space = 2

P = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6], [0.95, 0.05]])
P = P.reshape((2,2,2))

theta_b_1 = 0.1
theta_b_2 = 0.1
theta_b_3 = 0.95
theta_b_4 = 0.95
pi_b = np.array([[theta_b_1, 1-theta_b_1], [1-theta_b_2, theta_b_2], [theta_b_3, 1-theta_b_3], [1-theta_b_4, theta_b_4]])
pi_b = pi_b.reshape((2,2,2))
termination_space = 2

size_input = 1
zeta = 0.001

mu = np.array([0.5, 0.5])
max_epoch = 0

nTraj = 10 #100

std = np.empty(0)
aver_reward = np.empty(0)
TrainingSetDC = [[None]*1 for _ in range(10)]
LabelsDC = [[None]*1 for _ in range(10)]

for k in range(10):
    max_epoch = max_epoch + 50
    reward = np.empty(0)
    trajDC, controlDC, OptionDC, TerminationDC = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi, pi_lo, pi_b)
    dataDC = np.empty((0,1))
    labelsDC = np.empty((0))
    
    for i in range(nTraj):
        dataDC = np.append(dataDC, trajDC[i][:-1,:],0)
        labelsDC = np.append(labelsDC, controlDC[i])
        reward = np.append(reward, np.mean(trajDC[i]))
        
    aver_reward = np.append(aver_reward, np.mean(reward))
    std = np.append(std, np.std(reward))
    TrainingSetDC[k] = dataDC
    LabelsDC[k] = labelsDC

    
# %% Expert Performance

plt.figure()
plt.plot(np.linspace(len(TrainingSetDC[0]),len(TrainingSetDC[-1]),len(TrainingSetDC)), aver_reward, 'k', color='#CC4F1B')
plt.fill_between(np.linspace(len(TrainingSetDC[0]),len(TrainingSetDC[-1]),len(TrainingSetDC)), aver_reward-std, aver_reward+std,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.ylim(0,1.1)

# with open('Variables_saved/likelihood.npy', 'wb') as f:
#     np.save(f,[Likelihood, Model_orders])

Triple_expert = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

with open('Plots/Expert_performance.npy', 'wb') as f:
    np.save(f,[aver_reward, std, Triple_expert, TrainingSetDC, LabelsDC])
       
# %% initialization Learning for Batch BW INIT 1

init = 1

theta_hi_1 = 0.2
theta_hi_2 = 0.2

theta_lo_1 = 0.7
theta_lo_2 = 0.8
theta_lo_3 = 0.8
theta_lo_4 = 0.6

theta_b_1 = 0.2
theta_b_2 = 0.1
theta_b_3 = 0.8
theta_b_4 = 0.7

Triple_init1 = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

N=5 #Iterations
zeta = 0.001 #Failure factor

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

EV = hil.Experiment_design_discrete(LabelsDC, TrainingSetDC, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init1, gain_lambdas, gain_eta, 'discrete', max_epoch)

# %% HIL BATCH BW INIT 1

Options_list = []
Actions_list = []
Termination_list =[]
theta_hi_1_list = []
theta_hi_2_list = []

theta_lo_1_list = []
theta_lo_2_list = []
theta_lo_3_list = []
theta_lo_4_list = []

theta_b_1_list = []
theta_b_2_list = []
theta_b_3_list = []
theta_b_4_list = []

theta_hi_1_list.append(theta_hi_1)
theta_hi_2_list.append(theta_hi_2)

theta_lo_1_list.append(theta_lo_1)
theta_lo_2_list.append(theta_lo_2)
theta_lo_3_list.append(theta_lo_3)
theta_lo_4_list.append(theta_lo_4)

theta_b_1_list.append(theta_b_1)
theta_b_2_list.append(theta_b_2)
theta_b_3_list.append(theta_b_3)
theta_b_4_list.append(theta_b_4)

for i in range(len(TrainingSetDC)):
    EV.TrainingSet = TrainingSetDC[i] # check the parameters learnt for a different amount of training data
    EV.labels = LabelsDC[i]

    P_Termination, P_Actions, P_Options = hil.BaumWelch_discrete(EV)

    Termination_list.append(P_Termination)
    Actions_list.append(P_Actions)
    Options_list.append(P_Options)
    
    theta_hi_1_list.append(P_Options.theta_1)
    theta_hi_2_list.append(P_Options.theta_2)

    theta_lo_1_list.append(P_Actions.theta_1)
    theta_lo_2_list.append(P_Actions.theta_2)
    theta_lo_3_list.append(P_Actions.theta_3)
    theta_lo_4_list.append(P_Actions.theta_4)

    theta_b_1_list.append(P_Termination.theta_1)
    theta_b_2_list.append(P_Termination.theta_2)
    theta_b_3_list.append(P_Termination.theta_3)
    theta_b_4_list.append(P_Termination.theta_4)

# %% reprocess and store the parameters INIT 1

Parameters = [[None]*1 for _ in range(10)]

Theta_hi_1 = np.array(theta_hi_1_list)
Theta_hi_1 = Theta_hi_1[np.where(np.isnan(theta_hi_1_list) != True)[0]]
Parameters[0] = Theta_hi_1
Theta_hi_2 = np.array(theta_hi_2_list)
Theta_hi_2 = Theta_hi_2[np.where(np.isnan(theta_hi_2_list) != True)[0]]
Parameters[1]= Theta_hi_2
Theta_lo_1 = np.array(theta_lo_1_list)
Theta_lo_1 = Theta_lo_1[np.where(np.isnan(theta_lo_1_list) != True)[0]]
Parameters[2]= Theta_lo_1
Theta_lo_2 = np.array(theta_lo_2_list)
Theta_lo_2 = Theta_lo_2[np.where(np.isnan(theta_lo_2_list) != True)[0]]
Parameters[3]= Theta_lo_2
Theta_lo_3 = np.array(theta_lo_3_list)
Theta_lo_3 = Theta_lo_3[np.where(np.isnan(theta_lo_3_list) != True)[0]]
Parameters[4]= Theta_lo_3
Theta_lo_4 = np.array(theta_lo_4_list)
Theta_lo_4 = Theta_lo_4[np.where(np.isnan(theta_lo_4_list) != True)[0]]
Parameters[5]= Theta_lo_4
Theta_b_1 = np.array(theta_b_1_list)
Theta_b_1 = Theta_b_1[np.where(np.isnan(theta_b_1_list) != True)[0]]
Parameters[6]= Theta_b_1
Theta_b_2 = np.array(theta_b_2_list)
Theta_b_2 = Theta_b_2[np.where(np.isnan(theta_b_2_list) != True)[0]]
Parameters[7]= Theta_b_2
Theta_b_3 = np.array(theta_b_3_list)
Theta_b_3 = Theta_b_3[np.where(np.isnan(theta_b_3_list) != True)[0]]
Parameters[8]= Theta_b_3
Theta_b_4 = np.array(theta_b_4_list)
Theta_b_4 = Theta_b_4[np.where(np.isnan(theta_b_4_list) != True)[0]]
Parameters[9]= Theta_b_4

with open('Learnt_Parameters/learned_thetas_batch_BW_init{}.npy'.format(init), 'wb') as f:
    np.save(f,Parameters)

# %% initialization Learning for Batch BW INIT 2

init = 2

theta_hi_1 = 0.2
theta_hi_2 = 0.2

theta_lo_1 = 0.7
theta_lo_2 = 0.8
theta_lo_3 = 0.8
theta_lo_4 = 0.6

theta_b_1 = 0.5
theta_b_2 = 0.5
theta_b_3 = 0.5
theta_b_4 = 0.5

Triple_init1 = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

N=5 #Iterations
zeta = 0.001 #Failure factor

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

EV = hil.Experiment_design_discrete(LabelsDC, TrainingSetDC, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init1, gain_lambdas, gain_eta, 'discrete', max_epoch)

# %% HIL BATCH BW INIT 2

Options_list = []
Actions_list = []
Termination_list =[]
theta_hi_1_list = []
theta_hi_2_list = []

theta_lo_1_list = []
theta_lo_2_list = []
theta_lo_3_list = []
theta_lo_4_list = []

theta_b_1_list = []
theta_b_2_list = []
theta_b_3_list = []
theta_b_4_list = []

theta_hi_1_list.append(theta_hi_1)
theta_hi_2_list.append(theta_hi_2)

theta_lo_1_list.append(theta_lo_1)
theta_lo_2_list.append(theta_lo_2)
theta_lo_3_list.append(theta_lo_3)
theta_lo_4_list.append(theta_lo_4)

theta_b_1_list.append(theta_b_1)
theta_b_2_list.append(theta_b_2)
theta_b_3_list.append(theta_b_3)
theta_b_4_list.append(theta_b_4)

for i in range(len(TrainingSetDC)):
    EV.TrainingSet = TrainingSetDC[i]
    EV.labels = LabelsDC[i]

    P_Termination, P_Actions, P_Options = hil.BaumWelch_discrete(EV)

    Termination_list.append(P_Termination)
    Actions_list.append(P_Actions)
    Options_list.append(P_Options)
    
    theta_hi_1_list.append(P_Options.theta_1)
    theta_hi_2_list.append(P_Options.theta_2)

    theta_lo_1_list.append(P_Actions.theta_1)
    theta_lo_2_list.append(P_Actions.theta_2)
    theta_lo_3_list.append(P_Actions.theta_3)
    theta_lo_4_list.append(P_Actions.theta_4)

    theta_b_1_list.append(P_Termination.theta_1)
    theta_b_2_list.append(P_Termination.theta_2)
    theta_b_3_list.append(P_Termination.theta_3)
    theta_b_4_list.append(P_Termination.theta_4)

# %% reprocess and store the parameters INIT 1

Parameters = [[None]*1 for _ in range(10)]

Theta_hi_1 = np.array(theta_hi_1_list)
Theta_hi_1 = Theta_hi_1[np.where(np.isnan(theta_hi_1_list) != True)[0]]
Parameters[0] = Theta_hi_1
Theta_hi_2 = np.array(theta_hi_2_list)
Theta_hi_2 = Theta_hi_2[np.where(np.isnan(theta_hi_2_list) != True)[0]]
Parameters[1]= Theta_hi_2
Theta_lo_1 = np.array(theta_lo_1_list)
Theta_lo_1 = Theta_lo_1[np.where(np.isnan(theta_lo_1_list) != True)[0]]
Parameters[2]= Theta_lo_1
Theta_lo_2 = np.array(theta_lo_2_list)
Theta_lo_2 = Theta_lo_2[np.where(np.isnan(theta_lo_2_list) != True)[0]]
Parameters[3]= Theta_lo_2
Theta_lo_3 = np.array(theta_lo_3_list)
Theta_lo_3 = Theta_lo_3[np.where(np.isnan(theta_lo_3_list) != True)[0]]
Parameters[4]= Theta_lo_3
Theta_lo_4 = np.array(theta_lo_4_list)
Theta_lo_4 = Theta_lo_4[np.where(np.isnan(theta_lo_4_list) != True)[0]]
Parameters[5]= Theta_lo_4
Theta_b_1 = np.array(theta_b_1_list)
Theta_b_1 = Theta_b_1[np.where(np.isnan(theta_b_1_list) != True)[0]]
Parameters[6]= Theta_b_1
Theta_b_2 = np.array(theta_b_2_list)
Theta_b_2 = Theta_b_2[np.where(np.isnan(theta_b_2_list) != True)[0]]
Parameters[7]= Theta_b_2
Theta_b_3 = np.array(theta_b_3_list)
Theta_b_3 = Theta_b_3[np.where(np.isnan(theta_b_3_list) != True)[0]]
Parameters[8]= Theta_b_3
Theta_b_4 = np.array(theta_b_4_list)
Theta_b_4 = Theta_b_4[np.where(np.isnan(theta_b_4_list) != True)[0]]
Parameters[9]= Theta_b_4

with open('Learnt_Parameters/learned_thetas_batch_BW_init{}.npy'.format(init), 'wb') as f:
    np.save(f,Parameters)





