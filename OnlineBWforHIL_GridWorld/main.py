#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:02:52 2020

@author: vittorio
"""
import World 
import BatchBW_HIL 
import numpy as np

# %% Expert Policy Generation and simulation
expert = World.TwoRewards.Expert()
pi_hi, pi_lo, pi_b = expert.HierarchicalPolicy()
ExpertSim = expert.Simulation(pi_hi, pi_lo, pi_b)
max_epoch = 100
nTraj = 100
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardExpert)   
ExpertSim.HILVideoSimulation(controlExpert[best][:], trajExpert[best][:], 
                             OptionsExpert[best][:], psiExpert[best][:],"Videos/VideosExpert/sim_HierarchExpert.mp4")

# %% Batch BW for HIL with tabular parameterization
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space)

pi_hi_init_agent = Agent_BatchHIL.initialize_pi_hi()
pi_b_init_agent = Agent_BatchHIL.initialize_pi_b()
pi_lo_init_agent = Agent_BatchHIL.initialize_pi_lo()

pi_hi_agent = BatchBW_HIL.PI_HI(pi_hi_init_agent)
pi_b_agent = BatchBW_HIL.PI_B(pi_b_init_agent)
pi_lo_agent = BatchBW_HIL.PI_LO(pi_lo_init_agent)

alpha = Agent_BatchHIL.Alpha(pi_hi_agent.Policy , pi_b_agent.Policy , pi_lo_agent.Policy)
beta = Agent_BatchHIL.Beta(pi_hi_agent.Policy , pi_b_agent.Policy , pi_lo_agent.Policy)
gamma = Agent_BatchHIL.Gamma(alpha, beta)
gamma_tilde = Agent_BatchHIL.GammaTilde(alpha, beta, pi_hi_agent.Policy , pi_b_agent.Policy , pi_lo_agent.Policy)





