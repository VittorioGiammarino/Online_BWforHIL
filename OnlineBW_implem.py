#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:30:25 2020

@author: vittorio
"""

import numpy as np
import Simulation as sim
import matplotlib.pyplot as plt
import HierarchicalImitationLearning as hil

with open('Plots/Expert_performance.npy', 'rb') as g:
    Expert_aver_reward, Expert_std, Triple_Expert, TrainingSetDC, LabelsDC = np.load(g, allow_pickle = True)
    
# %% Initialization Learning for ONLINE BW INIT 1

mu = np.array([0.5, 0.5])
option_space = 2
action_space = 2
termination_space = 2
state_space = 2 
size_input = TrainingSetDC[0].shape[1]

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
max_epoch = 300

EV = hil.Experiment_design_discrete(LabelsDC[0], TrainingSetDC[0], size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init1, gain_lambdas, gain_eta, 'discrete', max_epoch)

# %% HIL ONLINE BW INIT 1

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

    T_min=len(EV.TrainingSet)/2
    P_Termination, P_Actions, P_Options = hil.Online_BaumWelch_discrete(EV, T_min, state_space)

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

with open('Learnt_Parameters/learned_thetas_online_BW_init{}.npy'.format(init), 'wb') as f:
    np.save(f,Parameters)

# %% initialization Learning for Online BW INIT 2

mu = np.array([0.5, 0.5])
option_space = 2
action_space = 2
termination_space = 2
state_space = 2 
size_input = TrainingSetDC[0].shape[1]

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
    EV.TrainingSet = TrainingSetDC[i] # check the parameters learnt for a different amount of training data
    EV.labels = LabelsDC[i]

    T_min=len(EV.TrainingSet)/2
    P_Termination, P_Actions, P_Options = hil.Online_BaumWelch_discrete(EV, T_min, state_space)

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

# %% reprocess and store the parameters INIT 2

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

with open('Learnt_Parameters/learned_thetas_online_BW_init{}.npy'.format(init), 'wb') as f:
    np.save(f,Parameters)




# %%
# max_epoch = 300
# nTraj = 100
# P = np.array([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.95, 0.05]])
# P = P.reshape((2,2,2))
# reward_online = np.empty(0)
# [trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi.P, pi_lo.P, pi_b.P)
# for k in range(nTraj):
#     reward_online = np.append(reward_online, np.mean(trajDC[k]))
# reward_onlineBW = np.mean(reward_online)

    
    
            
        
    



