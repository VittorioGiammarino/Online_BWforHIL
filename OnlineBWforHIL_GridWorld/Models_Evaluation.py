#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:59 2020

@author: vittorio
"""

import numpy as np

with open('Models/Saved_Model_Expert/pi_hi.npy', 'rb') as f:
    pi_hi_expert = np.load(f)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'rb') as f:
    pi_lo_expert = np.load(f)

with open('Models/Saved_Model_Expert/pi_b.npy', 'rb') as f:
    pi_b_expert = np.load(f)   
    
with open('Models/Saved_Model_Batch/pi_hi.npy', 'rb') as f:
    pi_hi_batch = np.load(f)
    
with open('Models/Saved_Model_Batch/pi_lo.npy', 'rb') as f:
    pi_lo_batch = np.load(f)

with open('Models/Saved_Model_Batch/pi_b.npy', 'rb') as f:
    pi_b_batch = np.load(f)   
    
with open('Models/Saved_Model_Online/pi_hi.npy', 'rb') as f:
    pi_hi_online = np.load(f)
    
with open('Models/Saved_Model_Online/pi_lo.npy', 'rb') as f:
    pi_lo_online = np.load(f)

with open('Models/Saved_Model_Online/pi_b.npy', 'rb') as f:
    pi_b_online = np.load(f)   
    