#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:28:40 2020

@author: vittorio
"""

import numpy as np
import Simulation as sim
import matplotlib.pyplot as plt
import HierarchicalImitationLearning as hil

# %%

init = 1

option_space=2
action_space = 2

P = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6], [0.95, 0.05]])
P = P.reshape((2,2,2))

size_input = 1
zeta = 0.001

mu = np.array([0.5, 0.5])
max_epoch = 300

nTraj = 100

std1 = np.empty(0)
aver_reward1 = np.empty(0)

with open('Learnt_Parameters/learned_thetas_batch_BW_init{}.npy'.format(init), 'rb') as f:
    Parameters1 = np.load(f)
    
for i in range(1,Parameters1.shape[1]):
    reward1 = np.empty(0)
    pi_hi, pi_lo, pi_b = hil.get_discrete_policy(Parameters1[:,i])
    [trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi.P, pi_lo.P, pi_b.P)
    
    for k in range(nTraj):
        reward1 = np.append(reward1, np.mean(trajDC[k]))
        
    aver_reward1 = np.append(aver_reward1, np.mean(reward1))
    std1 = np.append(std1, np.std(reward1))
# %%

init = 2

option_space=2
action_space = 2

P = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6], [0.95, 0.05]])
P = P.reshape((2,2,2))

size_input = 1
zeta = 0.001

mu = np.array([0.5, 0.5])
max_epoch = 300

nTraj = 100

std2 = np.empty(0)
aver_reward2 = np.empty(0)

with open('Learnt_Parameters/learned_thetas_batch_BW_init{}.npy'.format(init), 'rb') as f:
    Parameters2 = np.load(f)
    
for i in range(1,Parameters2.shape[1]):
    reward2 = np.empty(0)
    pi_hi, pi_lo, pi_b = hil.get_discrete_policy(Parameters2[:,i])
    [trajDC, controlDC, OptionDC, TerminationDC] = sim.Discrete_policy(zeta, mu, max_epoch, nTraj, option_space, action_space, size_input, P, pi_hi.P, pi_lo.P, pi_b.P)
    
    for k in range(nTraj):
        reward2 = np.append(reward2, np.mean(trajDC[k]))
        
    aver_reward2 = np.append(aver_reward2, np.mean(reward2))
    std2= np.append(std2, np.std(reward2))

    
# %%

with open('Plots/Expert_performance.npy', 'rb') as g:
    Expert_aver_reward, Expert_std, Triple_Expert = np.load(g, allow_pickle = True)

# %% Performance 

plt.figure()
plt.plot(np.linspace(20000,65000,10), Expert_aver_reward, 'k', color='#CC4F1B', label='Expert')
plt.fill_between(np.linspace(20000,65000,10), Expert_aver_reward-Expert_std, Expert_aver_reward+Expert_std,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.plot(np.linspace(20000,65000,10), aver_reward1, '--ok', color= '#3F7F4C', label = 'Agent2')
plt.fill_between(np.linspace(20000,65000,10), aver_reward1-std1, aver_reward2+std2, alpha=0.2, 
                 edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0.1)
plt.plot(np.linspace(20000,65000,10), aver_reward2, '-Dk', color= '#1B2ACC', label = 'Agent2')
plt.fill_between(np.linspace(20000,65000,10), aver_reward2-std2, aver_reward2+std2, alpha=0.2, 
                 edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0.1)
plt.ylim(0,1.1)
plt.legend()
plt.xlabel('Data samples')
plt.ylabel('Average Reward')
plt.savefig('Figures/Comparison_Expert_policy_learnt_reward.eps', format='eps')

# %% Theta evolution Init1
plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_hi_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[0,:], '--Dk', color= '#1B2ACC', label='$\\theta^{1}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_hi_2*np.ones(11), '--c', label='$\\theta^{2*}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[1,:], '--*m', label='$\\theta^{2}_{hi}$')
plt.ylim(0,1.1)
plt.legend(loc='upper center', fontsize='small', ncol=2)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{hi}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_hi_init1.eps', format='eps')

plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[2,:], '-dk', color= '#7cfc00', label='$\\theta^{1}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_2*np.ones(11), '--c', label='$\\theta^{2*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[3,:], '--om', label='$\\theta^{2}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_3*np.ones(11), '-ok', color='#1f77b4', label='$\\theta^{3*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[4,:], '-*r', label='$\\theta^{3}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_4*np.ones(11), '--g', label='$\\theta^{4*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[5,:], '--vb', label='$\\theta^{4}_{lo}$')
plt.ylim(0,1.1)
plt.legend(loc='lower center', fontsize='small', ncol=4)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{lo}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_lo_init1.eps', format='eps')

plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[6,:], '-Dk', color= '#7cfc00', label='$\\theta^{1}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_2*np.ones(11), '--c', label='$\\theta^{2*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[7,:], '-vm', label='$\\theta^{2}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_3*np.ones(11), '-ok', color='#1f77b4', label='$\\theta^{3*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[8,:], '--*r', label='$\\theta^{3}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_4*np.ones(11), '--g', label='$\\theta^{4*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters1[9,:], '--b', label='$\\theta^{4}_{b}$')
plt.ylim(0,1.1)
plt.legend(loc='middle center', fontsize='small', ncol=4)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{b}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_b_init1.eps', format='eps')

# %% Theta evolution Init2
plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_hi_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[0,:], '--Dk', color= '#1B2ACC', label='$\\theta^{1}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_hi_2*np.ones(11), '--c', label='$\\theta^{2*}_{hi}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[1,:], '--*m', label='$\\theta^{2}_{hi}$')
plt.ylim(0,1.1)
plt.legend(loc='upper center', fontsize='small', ncol=2)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{hi}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_hi_init2.eps', format='eps')

plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[2,:], '-dk', color= '#7cfc00', label='$\\theta^{1}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_2*np.ones(11), '--c', label='$\\theta^{2*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[3,:], '--om', label='$\\theta^{2}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_3*np.ones(11), '-ok', color='#1f77b4', label='$\\theta^{3*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[4,:], '-*r', label='$\\theta^{3}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_lo_4*np.ones(11), '--g', label='$\\theta^{4*}_{lo}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[5,:], '--vb', label='$\\theta^{4}_{lo}$')
plt.ylim(0,1.1)
plt.legend(loc='lower center', fontsize='small', ncol=4)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{lo}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_lo_init2.eps', format='eps')

plt.figure()
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_1*np.ones(11), '-ok', color='#CC4F1B', label='$\\theta^{1*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[6,:], '-Dk', color= '#7cfc00', label='$\\theta^{1}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_2*np.ones(11), '--c', label='$\\theta^{2*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[7,:], '-vm', label='$\\theta^{2}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_3*np.ones(11), '-ok', color='#1f77b4', label='$\\theta^{3*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[8,:], '--*r', label='$\\theta^{3}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Triple_Expert.theta_b_4*np.ones(11), '--g', label='$\\theta^{4*}_{b}$')
plt.plot(np.append(0,np.linspace(20000,65000,10)), Parameters2[9,:], '--b', label='$\\theta^{4}_{b}$')
plt.ylim(0,1.1)
plt.legend(loc='middle center', fontsize='small', ncol=4)
plt.xlabel('Data samples')
plt.ylabel('$\\theta_{b}$')
plt.savefig('Figures/Comparison_Expert_policy_learnt_theta_b_init2.eps', format='eps')


