#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:02:52 2020

@author: vittorio
"""

import World 

# %% 

expert = World.TwoRewards.Expert()
UTot, UR1, UR2, UBoth = expert.ComputeFlatPolicy()

