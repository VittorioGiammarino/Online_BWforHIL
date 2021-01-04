#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:27:04 2021

@author: vittorio
"""
import World 
import numpy as np

# %%

n_bins = (8, 8, 8, 8, 4, 4)
Q_table = np.zeros(n_bins + (2,))
expert = World.Acrobot.Expert.Expert_Q_learning(n_bins, Q_table)

Q_trained = expert.Training(10000)


# %%

with open('Models/Saved_Model_Expert/Q_trained.npy', 'wb') as f:
    np.save(f, Q_trained)

