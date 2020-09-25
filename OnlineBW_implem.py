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
    
# %%

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

EV = hil.Experiment_design_discrete(LabelsDC[-1], TrainingSetDC[-1], size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init1, gain_lambdas, gain_eta, 'discrete', max_epoch)

P_Options = hil.pi_hi_discrete(EV.Triple_init.theta_hi_1, EV.Triple_init.theta_hi_2)
P_Actions = hil.pi_lo_discrete(EV.Triple_init.theta_lo_1, EV.Triple_init.theta_lo_2,EV.Triple_init.theta_lo_3, EV.Triple_init.theta_lo_4)
P_Termination = hil.pi_b_discrete(EV.Triple_init.theta_b_1, EV.Triple_init.theta_b_2, EV.Triple_init.theta_b_3, EV.Triple_init.theta_b_4)

# %%

# Init
T_min=4900

zi = np.ones((option_space, termination_space, option_space, action_space, state_space, 1))
phi_h = np.ones((option_space, termination_space, option_space, action_space, state_space, termination_space, option_space,1))
norm = np.zeros((len(EV.mu), action_space, state_space))
P_option_given_obs = np.zeros((option_space, 1))

State = EV.TrainingSet[0].reshape(1,EV.size_input)
Action = EV.labels[0]

for a1 in range(action_space):
    for s1 in range(state_space):
        for o0 in range(option_space):
            for b1 in range(termination_space):
                for o1 in range(option_space):
                    state = s1*np.ones((1,1))
                    action = a1*np.ones((1,1))
                    zi[o0,b1,o1,a1,s1,0] = hil.Pi_combined(o1, o0, action, b1, P_Options.policy, P_Actions.policy, P_Termination.policy, 
                                                               state, zeta, option_space)
                                                       
            norm[o0,a1,s1]=EV.mu[o0]*np.sum(zi[:,:,:,a1,s1,0],(1,2))[o0]
            
        zi[:,:,:,a1,s1,0] = np.divide(zi[:,:,:,a1,s1,0],np.sum(norm[:,a1,s1]))
        if a1 == int(Action) and s1 == int(State):
            P_option_given_obs[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi[:,:,:,a1,s1,0],1))*EV.mu),0) 

for a1 in range(action_space):
    for s1 in range(state_space):
        for o0 in range(option_space):
            for b1 in range(termination_space):
                for o1 in range(option_space):
                    for bT in range(termination_space):
                        for oT in range(option_space):
                            if a1 == int(Action) and s1 == int(State):
                                phi_h[o0,b1,o1,a1,s1,bT,oT,0] = zi[o0,b1,o1,a1,s1,0]*EV.mu[o0]
                            else:
                                phi_h[o0,b1,o1,a1,s1,bT,oT,0] = 0.0001
            
for t in range(1,len(EV.TrainingSet)):
    
    #E-step
    zi_temp1 = np.ones((option_space, termination_space, option_space, action_space, state_space, 1))
    phi_h_temp = np.ones((option_space, termination_space, option_space, action_space, state_space, termination_space, option_space, 1))
    norm = np.zeros((len(EV.mu), action_space, state_space))
    P_option_given_obs_temp = np.zeros((option_space, 1))
    prod_term = np.ones((option_space, termination_space, option_space, action_space, state_space, termination_space, option_space))
    
    State = EV.TrainingSet[t].reshape(1,EV.size_input)
    Action = EV.labels[t]
    for at in range(action_space):
        for st in range(state_space):
            for ot_past in range(option_space):
                for bt in range(termination_space):
                    for ot in range(option_space):
                        state = s1*np.ones((1,1))
                        action = a1*np.ones((1,1))
                        zi_temp1[ot_past,bt,ot,at,st,0] = hil.Pi_combined(ot, ot_past, action, bt, P_Options.policy, 
                                                                          P_Actions.policy, P_Termination.policy, state, zeta, option_space)
                
                norm[ot_past,at,st] = P_option_given_obs[ot_past,t-1]*np.sum(zi_temp1[:,:,:,at,st,0],(1,2))[ot_past]
    
            zi_temp1[:,:,:,at,st,0] = np.divide(zi_temp1[:,:,:,at,st,0],np.sum(norm[:,at,st]))
            if at == int(Action) and st == int(State):
                P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,a1,s1,0],1))*P_option_given_obs[:,t-1]),0) 
            
    zi = np.concatenate((zi,zi_temp1),5)
    P_option_given_obs = np.concatenate((P_option_given_obs,P_option_given_obs_temp),1)
    
    zi_temp2 = np.ones((option_space, termination_space, option_space, action_space, state_space, 1))
    for at in range(action_space):
        for st in range(option_space):
            for ot_past in range(option_space):
                for bt in range(termination_space):
                    for ot in range(option_space):
                        for bT in range(termination_space):
                            for oT in range(option_space):
                                prod_term[ot_past, bt, ot, at, st, bT, oT] = np.sum(zi[:,bT,oT,int(Action),int(State),t]*np.sum(phi_h[ot_past,bt,ot,at,st,:,:,t-1],0))
                                if at == int(Action) and st == int(State):
                                    phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1/t)*zi[ot_past,bt,ot,at,st,t]*P_option_given_obs[ot_past,t-1] + (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                else:
                                    phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                    
    phi_h = np.concatenate((phi_h,phi_h_temp),7)

    #M-step    
    if t > T_min:
        theta_hi_1 = np.clip(np.divide(np.sum(phi_h[:,1,0,0,:,:,:,t]),np.sum(phi_h[:,1,:,0,:,:,:,t])),0,1)
        theta_hi_2 = np.clip(np.divide(np.sum(phi_h[:,1,1,1,:,:,:,t]),np.sum(phi_h[:,1,:,1,:,:,:,t])),0,1)
        theta_lo_1 = np.clip(np.divide(np.sum(phi_h[:,:,0,0,0,:,:,t]),np.sum(phi_h[:,:,0,0,:,:,:,t])),0,1)
        theta_lo_2 = np.clip(np.divide(np.sum(phi_h[:,:,1,0,1,:,:,t]),np.sum(phi_h[:,:,1,0,:,:,:,t])),0,1)
        theta_lo_3 = np.clip(np.divide(np.sum(phi_h[:,:,0,1,0,:,:,t]),np.sum(phi_h[:,:,0,1,:,:,:,t])),0,1)
        theta_lo_4 = np.clip(np.divide(np.sum(phi_h[:,:,1,1,1,:,:,t]),np.sum(phi_h[:,:,1,0,:,:,:,t])),0,1)
        theta_b_1 = np.clip(np.divide(np.sum(phi_h[0,0,:,0,:,:,:,t]),np.sum(phi_h[0,:,:,0,:,:,:,t])),0,1)
        theta_b_2 = np.clip(np.divide(np.sum(phi_h[1,1,:,0,:,:,:,t]),np.sum(phi_h[1,:,:,0,:,:,:,t])),0,1)
        theta_b_3 = np.clip(np.divide(np.sum(phi_h[0,0,:,1,:,:,:,t]),np.sum(phi_h[0,:,:,1,:,:,:,t])),0,1)
        theta_b_4 = np.clip(np.divide(np.sum(phi_h[1,1,:,1,:,:,:,t]),np.sum(phi_h[1,:,:,1,:,:,:,t])),0,1)
        
        P_Options = hil.pi_hi_discrete(theta_hi_1, theta_hi_2)
        P_Actions = hil.pi_lo_discrete(theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4)
        P_Termination = hil.pi_b_discrete(theta_b_1, theta_b_2, theta_b_3, theta_b_4)
                            




# %%
max_epoch = 300
nTraj = 100
P = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6], [0.95, 0.05]])
P = P.reshape((2,2,2))
reward1 = np.empty(0)
Theta = np.array([theta_hi_1,theta_hi_2,theta_lo_1,theta_lo_2,theta_lo_3,theta_lo_4,theta_b_1,theta_b_2,theta_b_3,theta_b_4])
pi_hi, pi_lo, pi_b = hil.get_discrete_policy(Theta)
[trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi.P, pi_lo.P, pi_b.P)
reward1 = np.mean(trajDC)

    
    
            
        
    



