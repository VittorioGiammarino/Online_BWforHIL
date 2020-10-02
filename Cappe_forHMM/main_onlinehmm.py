#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:42:05 2020

@author: vittorio
"""
import onlinehmm as ohmm
import numpy as np

# %%
#init
mu0 = np.array([-.5, .5])
q0 = np.array([[0.7, 0.3], [0.5, 0.5]])
v0 = 2
m = len(mu0)
    
mu = np.empty((0,m))
q = np.empty((m,m,0))
v = np.empty(0)
    
mu = np.append(mu,mu0.reshape((1,2)),0)
q = np.append(q,q0.reshape(m,m,1),2)
v = np.append(v,v0)
    
#generate data
Y, X = ohmm.generate_data(10001)
    
filt = ohmm.Initial_filter(mu0, Y[1], v0)
    
#auxiliary stoch init
T = np.zeros((m,m,m))
S0 = np.identity(m)
S1 = (Y[1])*np.identity(m)
S2 = (Y[1]**2)*np.identity(m)
    
n_min = 200
    
for i in range(0,len(Y)-2):
        
    gamma = 1/((i+1)**(0.6))
    q_new, mu_new, v_new, S0, S1, S2, filt, T = ohmm.update(Y[i+2], filt, T, S0, S1, S2, q[:,:,i], mu[i,:], v[i], 
                                                            gamma, n_min, i)
    mu = np.append(mu,mu_new.reshape((1,2)),0)
    q = np.append(q,q_new.reshape(m,m,1),2)
    v = np.append(v,v_new)
               
    
    