#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:02:52 2020

@author: vittorio
"""

import World 
import numpy as np

# %% 

expert = World.TwoRewards.Expert()
# UTot, UR1, UR2, UBoth = expert.ComputeFlatPolicy()

#%%
ss = expert.StateSpace()
pi_hi, pi_lo, pi_b = expert.HierarchicalPolicy()
expert.PlotHierachicalPolicy()

#%%
ExpertSim = expert.Simulation(pi_hi, pi_lo, pi_b)
max_epoch = 100
nTraj = 100

#%%
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, psiHIL, rewardHIL] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)
best = np.argmax(rewardHIL)   

# %%
ExpertSim.HILVideoSimulation(controlHIL[best][:],trajHIL[best][:],OptionsHIL[best][:], psiHIL[best][:],"Videos/VideosExpert/sim_HierarchExpert.mp4")


