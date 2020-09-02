#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:42:05 2020

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
    g = np.divide(np.exp(np.divide(-(Y*np.ones((m)) - mu)**2,2*v)),np.sqrt(v))
    
    # assuming uniform initial distribution
    c = np.sum(g)
    filt = g/c
    
    return filt


def update(Y, filt, T, S0, S1, S2, q, mu, v, gamma):
    
    # Auxiliary stoch init
    m = len(mu)
        
    #new likelyhood
    r = filt*q
    joint = r*np.divide(np.exp(np.divide(-(Y*np.ones((m)) - mu)**2,2*v)),np.sqrt(v))
    filt_sum = np.sum(joint,0)
    c = np.sum(filt)
    
    #normalize
    joint = joint/c
    filt = filt_sum/c
    
    # retrospective kernel
    temp = filt.reshape(1,m)
    norm_term = np.matmul(temp,q)
    r = r/norm_term
    
    #auxiliary matrix
    T1 = np.zeros((m,m,m))
    for k in range(m):
        T1[:,k,k] = r[:,k]
    
    # Statistics update
    T = gamma*T1 + (1-gamma)*np.matmul(T,r)
    S0 = gamma*np.identity(m) + (1-gamma)*np.matmul(S0,r)
    S1 = gamma*np.identity(m)*Y + (1-gamma)*np.matmul(S1,r)
    S2 = gamma*np.identity(m)*(Y**2) + (1-gamma)*np.matmul(S2,r)
    
    #M-step auxiliary
    temp = filt.reshape(m,1)
    T_a = np.matmul(T,temp)
    T_a = T_a.reshape((m,m))
    S0_a = np.matmul(S0,temp)
    S1_a = np.matmul(S1,temp)
    S2_a = np.matmul(S2,temp)
    
    #M-step 
    q = T_a/np.sum(T_a,1)
    mu = S1_a/S0_a
    v = np.divide(np.sum(S2_a - (mu**2)*S0_a),np.sum(S0_a))

def generate_data():
    
    
    
def main():
    mu = np.array([-.5, .5])
    q = np.array([[0.7, 0.5], [-0.5, 0.5]])
    v = 2
    
    #auxiliary stoch init
    m = len(mu)
    T = np.zeros((m,m,m))
    S0 = np.identity(m)
    S1 = (Y)*np.identity(m)
    S2 = (Y**2)*np.identity(m)
    
    