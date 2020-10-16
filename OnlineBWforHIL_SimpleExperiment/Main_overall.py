#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:21:28 2020

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

P = np.array([[0.9, 0.1], [0.5, 0.5], [0.9, 0.1], [0.95, 0.05]])
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

nTraj = 1
max_epoch = 1000
trajDC, controlDC, OptionDC, TerminationDC = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi, pi_lo, pi_b)
dataDC = np.empty((0,1))
labelsDC = np.empty((0))
    
for i in range(nTraj):
    dataDC = np.append(dataDC, trajDC[i][:-1,:],0)
    labelsDC = np.append(labelsDC, controlDC[i])
        
TrainingSetDC = dataDC
LabelsDC = labelsDC
Reward_expert = np.sum(TrainingSetDC)/len(TrainingSetDC)    

Triple_expert = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

# %% initialization Learning for Batch BW INIT 1

init = 1

theta_hi_1 = 0.05
theta_hi_2 = 0.05

theta_lo_1 = 0.8
theta_lo_2 = 0.9
theta_lo_3 = 0.95
theta_lo_4 = 0.8

theta_b_1 = 0.1
theta_b_2 = 0.1
theta_b_3 = 0.95
theta_b_4 = 0.95

Triple_init1 = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

N=5 #Iterations
zeta = 0.001 #Failure factor

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

EV = hil.Experiment_design_discrete(LabelsDC, TrainingSetDC, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init1, gain_lambdas, gain_eta, 'discrete', max_epoch)

# %% HIL BATCH BW 

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

# %% evaluation Batch
[trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, P_Options.P, P_Actions.P, P_Termination.P)    
Reward_agent_BatchBW = np.sum(trajDC)/len(trajDC[0])

# %% Online BW for HIL
state_space = 2 
T_min=len(EV.TrainingSet)/2

P_Termination_Online, P_Actions_Online, P_Options_Online = hil.Online_BaumWelch_discrete(EV, T_min, state_space)
    
theta_hi_1_list.append(P_Options_Online.theta_1)
theta_hi_2_list.append(P_Options_Online.theta_2)

theta_lo_1_list.append(P_Actions_Online.theta_1)
theta_lo_2_list.append(P_Actions_Online.theta_2)
theta_lo_3_list.append(P_Actions_Online.theta_3)
theta_lo_4_list.append(P_Actions_Online.theta_4)

theta_b_1_list.append(P_Termination_Online.theta_1)
theta_b_2_list.append(P_Termination_Online.theta_2)
theta_b_3_list.append(P_Termination_Online.theta_3)
theta_b_4_list.append(P_Termination_Online.theta_4)

# %% evaluation Online

[trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, 
                                                                   P_Options_Online.P, P_Actions_Online.P, P_Termination_Online.P)

Reward_agent_OnlineBW = np.sum(trajDC)/len(trajDC[0])

