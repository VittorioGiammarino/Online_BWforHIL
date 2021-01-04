#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:27:04 2021

@author: vittorio
"""
import World 
import numpy as np

# %%
n_bins = (3, 3, 6, 6)
Q_table = np.zeros(n_bins + (2,))
expert = World.CartPole.Expert(n_bins, 1, 1, Q_table)
lower_bounds = [expert.env.observation_space.low[0], -0.2, expert.env.observation_space.low[2], -np.radians(50) ]
upper_bounds = [expert.env.observation_space.high[0], 0.2, expert.env.observation_space.high[2], np.radians(50) ]
expert = World.CartPole.Expert(n_bins, lower_bounds, upper_bounds, Q_table)

Q_trained = expert.Training(10000)


# %%

with open('Models/Saved_Model_Expert/Q_trained.npy', 'wb') as f:
    np.save(f, Q_trained)

