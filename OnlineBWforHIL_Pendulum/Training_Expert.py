#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:27:04 2021

@author: vittorio
"""
import World 
import numpy as np

# %%
W = World.Pendulum.Expert.train()

with open('Models/Saved_Model_Expert/W_weights.npy', 'wb') as f:
    np.save(f, W)


