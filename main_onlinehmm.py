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


def update(Y, filt, T, S0, S1, S2, q, mu, v, gamma,n_min,i):
    
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
        
    if i>n_min:
    
        #M-step auxiliary
        temp = filt.reshape(m,1)
        T_a = np.matmul(T,temp)
        T_a = T_a.reshape((m,m))
        S0_a = np.matmul(S0,temp)
        S1_a = np.matmul(S1,temp)
        S2_a = np.matmul(S2,temp)
        
        #M-step 
        q_up = np.transpose(np.transpose(T_a)/np.sum(T_a,1))
        mu_up = S1_a/S0_a
        v_up = np.divide(np.sum(S2_a - (mu**2)*S0_a),np.sum(S0_a))
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
        

def main():
    
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
    Y, X = generate_data(10001)
    
    filt = Initial_filter(mu0, Y[1], v0)
    
    #auxiliary stoch init
    T = np.zeros((m,m,m))
    S0 = np.identity(m)
    S1 = (Y[1])*np.identity(m)
    S2 = (Y[1]**2)*np.identity(m)
    
    n_min = 200
    
    
    for i in range(0,len(Y)-2):
        
        gamma = 1/(i+1)
        q_new, mu_new, v_new, S0, S1, S2, filt, T = update(Y[i+2], filt, T, S0, S1, S2, q[:,:,i], mu[i,:], v[i], 
                                                           gamma, n_min, i)
        mu = np.append(mu,mu_new.reshape((1,2)),0)
        q = np.append(q,q_new.reshape(m,m,1),2)
        v = np.append(v,v_new)
               
    
    