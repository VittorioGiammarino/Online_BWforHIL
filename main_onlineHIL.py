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

# %%
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
max_epoch = 100

nTraj = 300

[trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi, pi_lo, pi_b)

# %%

sample_traj = 0

fig = plt.figure()
ax1 = plt.subplot(411)
plot_state = plt.plot(np.linspace(0,len(trajDC[sample_traj]), len(trajDC[sample_traj])), trajDC[sample_traj])
#plt.xlabel('step')
plt.ylabel('state')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412, sharex=ax1)
plot_action = plt.plot(np.linspace(0,len(trajDC[sample_traj])-1, len(trajDC[sample_traj])-1), controlDC[sample_traj])
#plt.xlabel('step')
plt.ylabel('action')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(413, sharex=ax1)
plot_option = plt.plot(np.linspace(0,len(trajDC[sample_traj]), len(trajDC[sample_traj])), OptionDC[sample_traj][1:])
#plt.xlabel('step')
plt.ylabel('option')
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = plt.subplot(414, sharex=ax1)
plot_termination = plt.plot(np.linspace(0,len(trajDC[sample_traj]), len(trajDC[sample_traj])), TerminationDC[sample_traj])
plt.xlabel('step')
plt.ylabel('termination')
plt.show()

# %%

dataDC = np.empty((0,1))
labelsDC = np.empty((0))

for i in range(len(trajDC)):
    dataDC = np.append(dataDC, trajDC[i][:-1,:],0)
    labelsDC = np.append(labelsDC, controlDC[i])

TrainingSetDC = dataDC
LabelsDC = labelsDC
   
# %% initialization

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

Triple_init = hil.Triple_discrete(theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4)

N=5 #Iterations
zeta = 0.001 #Failure factor

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

EV = hil.Experiment_design_discrete(labelsDC, TrainingSetDC, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init, gain_lambdas, gain_eta, 'discrete', max_epoch)

# %% HIL

P_Options = hil.pi_hi_discrete(EV.Triple_init.theta_hi_1, EV.Triple_init.theta_hi_2)
P_Actions = hil.pi_lo_discrete(EV.Triple_init.theta_lo_1, EV.Triple_init.theta_lo_2,EV.Triple_init.theta_lo_3, EV.Triple_init.theta_lo_4)
P_Termination = hil.pi_b_discrete(EV.Triple_init.theta_b_1, EV.Triple_init.theta_b_2, EV.Triple_init.theta_b_3, EV.Triple_init.theta_b_4)

state_0_index = np.where(EV.TrainingSet[:,0] == 0)[0]
state_1_index = np.where(EV.TrainingSet[:,0] == 1)[0]

action_0_index = np.where(EV.labels[:]==0)[0]
action_1_index = np.where(EV.labels[:]==1)[0]
action_0_state_0_index = hil.match_vectors(action_0_index, state_0_index)
action_1_state_0_index = hil.match_vectors(action_1_index, state_0_index)
action_0_state_1_index = hil.match_vectors(action_0_index, state_1_index)
action_1_state_1_index = hil.match_vectors(action_1_index, state_1_index)
    
for n in range(EV.N):
    print('iter', n+1, '/', EV.N)
        
    alpha = hil.Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                      EV.zeta, P_Options.policy, P_Actions.policy, P_Termination.policy)
    beta = hil.Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                    P_Options.policy, P_Actions.policy, P_Termination.policy)
    gamma = hil.Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
    gamma_tilde = hil.GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
                                 P_Options.policy, P_Actions.policy, P_Termination.policy, EV.zeta, EV.option_space, EV.termination_space)
    
    # M step
    gamma_state_0 = gamma[:,:,state_0_index]
    gamma_state_1 = gamma[:,:,state_1_index]
    theta_hi_1 = np.clip(np.divide(np.sum(gamma_state_0[0,1,:]),np.sum(gamma_state_0[:,1,:])),0,1)
    theta_hi_2 = np.clip(np.divide(np.sum(gamma_state_1[1,1,:]),np.sum(gamma_state_1[:,1,:])),0,1)
    
    theta_lo_1 = np.clip(np.divide(np.sum(gamma[0,:,action_0_state_0_index]),np.sum(gamma_state_0[0,:,:])),0,1)
    theta_lo_2 = np.clip(np.divide(np.sum(gamma[1,:,action_1_state_0_index]),np.sum(gamma_state_0[1,:,:])),0,1)
    theta_lo_3 = np.clip(np.divide(np.sum(gamma[0,:,action_0_state_1_index]),np.sum(gamma_state_1[0,:,:])),0,1)
    theta_lo_4 = np.clip(np.divide(np.sum(gamma[1,:,action_1_state_1_index]),np.sum(gamma_state_1[1,:,:])),0,1)
    
    gamma_tilde_state_0 = gamma_tilde[:,:,state_0_index]
    gamma_tilde_state_1 = gamma_tilde[:,:,state_1_index]
    theta_b_1 = np.clip(np.divide(np.sum(gamma_tilde_state_0[0,0,:]),np.sum(gamma_tilde_state_0[0,:,:])),0,1)
    theta_b_2 = np.clip(np.divide(np.sum(gamma_tilde_state_0[1,1,:]),np.sum(gamma_tilde_state_0[1,:,:])),0,1)
    theta_b_3 = np.clip(np.divide(np.sum(gamma_tilde_state_1[0,0,:]),np.sum(gamma_tilde_state_1[0,:,:])),0,1)
    theta_b_4 = np.clip(np.divide(np.sum(gamma_tilde_state_1[1,1,:]),np.sum(gamma_tilde_state_1[1,:,:])),0,1)
    
    P_Options = hil.pi_hi_discrete(theta_hi_1, theta_hi_2)
    P_Actions = hil.pi_lo_discrete(theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4)
    P_Termination = hil.pi_b_discrete(theta_b_1, theta_b_2, theta_b_3, theta_b_4)

# %% Test

nTraj = 300

[trajDC_l, controlDC_l, OptionDC_l, TerminationDC_l] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, 
                                                                          P_Options.P, P_Actions.P, P_Termination.P)
# %%

sample_traj = 0

fig = plt.figure()
ax1 = plt.subplot(411)
plot_state = plt.plot(np.linspace(0,len(trajDC_l[sample_traj]), len(trajDC_l[sample_traj])), trajDC_l[sample_traj])
#plt.xlabel('step')
plt.ylabel('state')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412, sharex=ax1)
plot_action = plt.plot(np.linspace(0,len(trajDC_l[sample_traj])-1, len(trajDC_l[sample_traj])-1), controlDC_l[sample_traj])
#plt.xlabel('step')
plt.ylabel('action')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(413, sharex=ax1)
plot_option = plt.plot(np.linspace(0,len(trajDC_l[sample_traj]), len(trajDC_l[sample_traj])), OptionDC_l[sample_traj][1:])
#plt.xlabel('step')
plt.ylabel('option')
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = plt.subplot(414, sharex=ax1)
plot_termination = plt.plot(np.linspace(0,len(trajDC_l[sample_traj]), len(trajDC_l[sample_traj])), TerminationDC_l[sample_traj])
plt.xlabel('step')
plt.ylabel('termination')
plt.show()



