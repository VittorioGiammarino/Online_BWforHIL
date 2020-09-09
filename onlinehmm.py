#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:35:21 2020

@author: vittorio
"""


import numpy as np

# implementation of Online Baum-Welch for gaussian distribution (cappe 2011)

def Initial_filter(mu, Y, v):
    
    # inputs:
        # Y initial observation (scalar)
        # mu means of the m states (m x 1)
        # v variance scalar
    
    m = len(mu)
    
    #likelihood factor
    g = np.divide(np.exp(np.divide(-0.5*(Y*np.ones((m)) - mu)**2,v)),np.sqrt(v))
    
    # assuming uniform initial distribution
    c = np.sum(g)
    filt = g/c
    
    return filt


def update(Y, filt, T, S0, S1, S2, q, mu, v, gamma, n_min, i):
    
    # Auxiliary stoch init
    m = len(mu)
        
    #new likelyhood
    r = np.transpose(filt*np.transpose(q))
    joint = r*(np.ones((m,m))*np.divide(np.exp(np.divide(-(Y*np.ones((m)) - mu)**2,2*v)),np.sqrt(v)))
    filt_sum = np.sum(joint,0)
    c = np.sum(filt_sum)
    
    #normalize
    joint = joint/c
    filt = filt_sum/c
    
    # retrospective kernel
    temp = np.sum(r,0)
    norm_term = np.ones((m,m))*temp
    r = r/norm_term
    
    #auxiliary matrix
    T1 = np.zeros((m,m,m))
    for k in range(m):
        T1[:,k,k] = r[:,k]
        
    # Statistics update
    T_temp = np.matmul(T.reshape(m*m,m),r)
    T = gamma*T1 + (1-gamma)*T_temp.reshape((m,m,m))
    S0 = gamma*np.identity(m) + (1-gamma)*np.matmul(S0,r)
    S1 = gamma*np.identity(m)*Y + (1-gamma)*np.matmul(S1,r)
    S2 = gamma*np.identity(m)*(Y**2) + (1-gamma)*np.matmul(S2,r)
        
    if i>n_min:
        #M-step auxiliary
        temp = np.matmul(T.reshape(m*m,m),filt)
        T_a = temp.reshape(m,m)
        T_a = T_a.reshape((m,m))
        S0_a = np.matmul(S0,filt)
        S1_a = np.matmul(S1,filt)
        S2_a = np.matmul(S2,filt)
        
        #M-step 
        q_up = np.transpose(np.transpose(T_a)/np.sum(T_a,1))
        mu_up = S1_a/S0_a
        v_up = np.divide(np.sum(S2_a - (mu_up**2)*S0_a),np.sum(S0_a))
    else:
        q_up = q
        mu_up = mu
        v_up = v
        
    return q_up, mu_up, v_up, S0, S1, S2, filt, T


def generate_data(n_data):
    q_star = np.array([[0.95, 0.05], [0.3, 0.7]])
    mu_star = np.array([0, 1])
    v_star = 0.5
    
    x = np.ones(1)
    y = np.zeros(1)
    for k in range(n_data):
        prob_x = q_star[int(x[k]),:]
        prob_x = prob_x.reshape(1,len(mu_star))
        prob_x_rescaled = np.divide(prob_x,np.amin(prob_x)+0.01)
        for i in range(1,prob_x_rescaled.shape[1]):
            prob_x_rescaled[0,i]=prob_x_rescaled[0,i]+prob_x_rescaled[0,i-1]
        draw_x = np.divide(np.random.rand(),np.amin(prob_x)+0.01)
        x = np.append(x,np.amin(np.where(draw_x<prob_x_rescaled[0,:])))
        
        y = np.append(y,np.random.normal(mu_star[int(x[k+1])], np.sqrt(v_star)))
        
    return y,x
        