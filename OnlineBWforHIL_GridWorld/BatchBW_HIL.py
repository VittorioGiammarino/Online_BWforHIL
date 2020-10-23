#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:58:25 2020

@author: vittorio
"""

import numpy as np
import World

def ProcessData(traj,control,psi,stateSpace):
    Xtr = np.empty((1,0),int)
    inputs = np.empty((3,0),int)

    for i in range(len(traj)):
        Xtr = np.append(Xtr, control[i][:])
        inputs = np.append(inputs, np.transpose(np.concatenate((stateSpace[traj[i][:-1],:], psi[i][:-1].reshape(len(psi[i])-1,1)),1)), axis=1) 
    
    labels = Xtr.reshape(len(Xtr),1)
    TrainingSet = np.transpose(inputs) 
    
    return labels, TrainingSet

class PI_LO:
    def __init__(self, pi_lo):
        self.pi_lo = pi_lo
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, state, option):
        stateID = self.expert.FindStateIndex(state)
        prob_distribution = self.pi_lo[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_B:
    def __init__(self, pi_b):
        self.pi_b = pi_b
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, state, option):
        stateID = self.expert.FindStateIndex(state)
        prob_distribution = self.pi_b[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_HI:
    def __init__(self, pi_hi):
        self.pi_hi = pi_hi
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, state):
        stateID = self.expert.FindStateIndex(state)
        prob_distribution = self.pi_hi[stateID,:]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
    

class BatchHIL:
    def __init__(self, TrainingSet, Labels, option_space):
        self.TrainingSet = TrainingSet
        self.Labels = Labels
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.action_space = int(np.max(Labels)+1)
        self.termination_space = 2
        self.zeta = 0.0001
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        self.environment = World.TwoRewards.Environment()
        self.expert = World.TwoRewards.Expert()
        
        
    def match_vectors(vector1,vector2):
    
        result = np.empty((0),int)
    
        for i in range(len(vector1)):
            for j in range(len(vector2)):
                if vector1[i]==vector2[j]:
                    result = np.append(result, int(vector1[i]))
                
        return result
        
    def FindStateIndex(self, value):
            
        stateSpace = self.expert.StateSpace()
        K = stateSpace.shape[0];
        stateIndex = 0
    
        for k in range(0,K):
            if stateSpace[k,0]==value[0,0] and stateSpace[k,1]==value[0,1] and stateSpace[k,2]==value[0,2]:
                stateIndex = k
    
        return stateIndex
        

    def initialize_pi_hi(self):
        stateSpace = self.expert.StateSpace()
        pi_hi = np.empty((0,self.option_space))
        for i in range(len(stateSpace)):
            prob_temp = np.random.uniform(0,1,self.option_space)
            prob_temp = np.divide(prob_temp, np.sum(prob_temp)).reshape(1,len(prob_temp))
            pi_hi = np.append(pi_hi, prob_temp, axis=0)
            
        return pi_hi
    
    def initialize_pi_lo(self):
        stateSpace = self.expert.StateSpace()
        pi_lo = np.zeros((len(stateSpace),self.action_space, self.option_space))
        for i in range(len(stateSpace)):
            for o in range(self.option_space):
                prob_temp = np.random.uniform(0,1,self.action_space)
                prob_temp = np.divide(prob_temp, np.sum(prob_temp))
                pi_lo[i,:,o] = prob_temp
                
        return pi_lo
    
    def initialize_pi_b(self):
        stateSpace = self.expert.StateSpace()
        pi_b = np.zeros((len(stateSpace),self.termination_space, self.option_space))
        for i in range(len(stateSpace)):
            for o in range(self.option_space):
                prob_temp = np.random.uniform(0,1,self.termination_space)
                prob_temp = np.divide(prob_temp, np.sum(prob_temp))
                pi_b[i,:,o] = prob_temp
                
        return pi_b
    
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = BatchHIL.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
        else:
            o_prob_tilde = np.divide(zeta,option_space)
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state, ot):
        Pi_lo = Pi_lo_parameterization(state, ot)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state, ot):
        Pi_b = Pi_b_parameterization(state, ot)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = np.clip(BatchHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(BatchHIL.Pi_lo(a, Pi_lo_parameterization, state, ot),0.0001,1)
        Pi_b_eval = np.clip(BatchHIL.Pi_b(b, Pi_b_parameterization, state, ot_past),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
    
    def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                         Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()= [option_space, termination_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = BatchHIL.Pi_combined(ot, ot_past, a, bt, 
                                                       Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                                       state, zeta, option_space)
                alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
        alpha = np.divide(alpha,np.sum(alpha))
            
        return alpha
    
    def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()=[option_space, termination_space]
        #   mu is the initial distribution over options: mu.shape()=[1,option_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = BatchHIL.Pi_combined(ot, ot_past, a, bt, 
                                                            Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                                            state, zeta, option_space)
                    alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
        alpha = np.divide(alpha, np.sum(alpha))
            
        return alpha

    def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     beta is the backward message: beta.shape()= [option_space, termination_space]
        # =============================================================================
        beta = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                for i1_next in range(option_space):
                    ot_next = i1_next
                    for i2_next in range(termination_space):
                        if i2 == 1:
                            b_next=True
                        else:
                            b_next=False
                        beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*BatchHIL.Pi_combined(ot_next, ot, a, b_next, 
                                                                                                    Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                                    Pi_b_parameterization, state, zeta, option_space)
        beta = np.divide(beta,np.sum(beta))
    
        return beta

    def Alpha(self, pi_hi, pi_b, pi_lo):
        alpha = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('alpha iter', t+1, '/', len(self.TrainingSet))
            if t ==0:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.Labels[t]
                alpha[:,:,t] = BatchHIL.ForwardFirstRecursion(self.mu, action, pi_hi, 
                                                              pi_lo, pi_b, state, self.zeta, self.option_space, self.termination_space)
            else:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.Labels[t]
                alpha[:,:,t] = BatchHIL.ForwardRecursion(alpha[:,:,t-1], action, pi_hi, 
                                                        pi_lo, pi_b, 
                                                        state, self.zeta, self.option_space, self.termination_space)
           
        return alpha

    def Beta(self, pi_hi, pi_b, pi_lo):
        beta = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        beta[:,:,len(self.TrainingSet)-1] = np.divide(np.ones((self.option_space,self.termination_space)),2*self.option_space)
    
        for t_raw in range(len(self.TrainingSet)-1):
            t = len(self.TrainingSet) - (t_raw+1)
            print('beta iter', t_raw+1, '/', len(self.TrainingSet)-1)
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.Labels[t]
            beta[:,:,t-1] = BatchHIL.BackwardRecursion(beta[:,:,t], action, pi_hi, 
                                                       pi_lo, pi_b, state, self.zeta, 
                                                       self.option_space, self.termination_space)
        
        return beta

    def Smoothing(option_space, termination_space, alpha, beta):
        gamma = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot=i1
            for i2 in range(termination_space):
                gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
            gamma = np.divide(gamma,np.sum(gamma))
    
        return gamma

    def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                    Pi_b_parameterization, state, zeta, option_space, termination_space):
        gamma_tilde = np.empty((option_space, termination_space))
        for i1_past in range(option_space):
            ot_past = i1_past
            for i2 in range(termination_space):
                if i2 == 1:
                    b=True
                else:
                    b=False
                for i1 in range(option_space):
                    ot = i1
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*BatchHIL.Pi_combined(ot, ot_past, a, b, 
                                                                                                         Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                                         Pi_b_parameterization, state, zeta, option_space)
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
        gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
        return gamma_tilde

    def Gamma(self, alpha, beta):
        gamma = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('gamma iter', t+1, '/', len(self.TrainingSet))
            gamma[:,:,t]=BatchHIL.Smoothing(self.option_space, self.termination_space, alpha[:,:,t], beta[:,:,t])
        
        return gamma

    def GammaTilde(self, alpha, beta, pi_hi, pi_b, pi_lo):
        gamma_tilde = np.zeros((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(1,len(self.TrainingSet)):
            print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.Labels[t]
            gamma_tilde[:,:,t]=BatchHIL.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                        pi_hi, pi_lo, pi_b, 
                                                        state, self.zeta, self.option_space, self.termination_space)
        return gamma_tilde
            
        
