#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:29:37 2020

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
                
    def Policy(self, stateID, option):
        prob_distribution = self.pi_lo[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_B:
    def __init__(self, pi_b):
        self.pi_b = pi_b
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, stateID, option):
        prob_distribution = self.pi_b[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_HI:
    def __init__(self, pi_hi):
        self.pi_hi = pi_hi
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, stateID):
        prob_distribution = self.pi_hi[stateID,:]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution

class OnlineHIL:
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
        
    def FindStateIndex(self, value):
        stateSpace = np.unique(self.TrainingSet, axis=0) # NOTE THIS IS REALLY IMPORTANT
        K = stateSpace.shape[0];
        stateIndex = 0
    
        for k in range(0,K):
            if stateSpace[k,0]==value[0,0] and stateSpace[k,1]==value[0,1] and stateSpace[k,2]==value[0,2]:
                stateIndex = k
    
        return stateIndex
    
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = OnlineHIL.Pi_hi(ot, Pi_hi_parameterization, state)
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
        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(OnlineHIL.Pi_lo(a, Pi_lo_parameterization, state, ot),0.0001,1)
        Pi_b_eval = np.clip(OnlineHIL.Pi_b(b, Pi_b_parameterization, state, ot_past),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
        
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
    
    def TrainingSetID(self):
        TrainingSetID = np.empty((0,1))
        for i in range(len(self.TrainingSet)):
            ID = OnlineHIL.FindStateIndex(self,self.TrainingSet[i,:].reshape(1,self.size_input))
            TrainingSetID = np.append(TrainingSetID, [[ID]], axis=0)
            
        return TrainingSetID
        
    def UpdatePiHi(self, Old_pi_hi, phi):
        New_pi_hi = np.zeros((Old_pi_hi.shape[0], Old_pi_hi.shape[1]))
        stateSpace = self.expert.StateSpace()
        stateSpace_phi = np.unique(self.TrainingSet, axis=0)
        stateID_phi = 0
        temp_theta = np.zeros((1,self.option_space))
            
        for stateID in range(len(stateSpace)):
            state = stateSpace[stateID,:]
            if stateID_phi == len(stateSpace_phi):
                stateID_phi = 0
            state_phi = stateSpace_phi[stateID_phi,:]
            match = state==state_phi
            for option in range(self.option_space):
                
                if not match.all():
                    temp_theta[0,option] = Old_pi_hi[stateID,option]
                elif np.sum(phi[:,1,:,stateID_phi,:,0])==0:
                    temp_theta[0,option] = Old_pi_hi[stateID,option]
                else:
                    temp_theta[0,option] = np.clip(np.divide(np.sum(phi[:,1,option,stateID_phi,:,0]),np.sum(phi[:,1,:,stateID_phi,:,0])),0,1) 

            if match.all():
                stateID_phi = stateID_phi + 1
                    
            temp_theta = np.divide(temp_theta, np.sum(temp_theta))
            New_pi_hi[stateID,:] = temp_theta
        
        return New_pi_hi
    
    def UpdatePiLo(self, Old_pi_lo, phi):
        New_pi_lo = np.zeros((Old_pi_lo.shape[0], Old_pi_lo.shape[1], Old_pi_lo.shape[2]))
        stateSpace = self.expert.StateSpace()
        stateSpace_phi = np.unique(self.TrainingSet, axis=0)
        stateID_phi = 0
        temp_theta = np.zeros((1,self.action_space))
        
        for option in range(self.option_space):
            for stateID in range(len(stateSpace)):
                state = stateSpace[stateID,:]
                if stateID_phi == len(stateSpace_phi):
                    stateID_phi = 0
                state_phi = stateSpace_phi[stateID_phi,:]
                match = state==state_phi
                for action in range(self.action_space):
                    if not match.all():
                        temp_theta[0,action] = Old_pi_lo[stateID,action,option]
                    elif np.sum(phi[:,:,option,stateID_phi,:,0])==0:
                        temp_theta[0,action] = Old_pi_lo[stateID,action,option]
                    else:
                        temp_theta[0,action] = np.clip(np.divide(np.sum(phi[:,:,option,stateID_phi,action,0]),np.sum(phi[:,:,option,stateID_phi,:,0])),0,1)
                        
                if match.all():
                    stateID_phi = stateID_phi + 1
                    
                temp_theta = np.divide(temp_theta, np.sum(temp_theta))
                New_pi_lo[stateID,:,option] = temp_theta
        
        return New_pi_lo
    
    def UpdatePiB(self, Old_pi_b, phi):
        New_pi_b = np.zeros((Old_pi_b.shape[0], Old_pi_b.shape[1], Old_pi_b.shape[2]))
        stateSpace = self.expert.StateSpace()
        stateSpace_phi = np.unique(self.TrainingSet, axis=0)
        stateID_phi = 0
        temp_theta = np.zeros((1,self.termination_space))
            
        for option in range(self.option_space):
            for stateID in range(len(stateSpace)):
                state = stateSpace[stateID,:]
                if stateID_phi == len(stateSpace_phi):
                    stateID_phi = 0
                state_phi = stateSpace_phi[stateID_phi,:]
                match = state==state_phi
                for termination_boolean in range(self.termination_space):
                    if not match.all():
                        temp_theta[0,termination_boolean] = Old_pi_b[stateID,termination_boolean,option]
                    elif np.sum(phi[option,:,:,stateID_phi,:,0])==0:
                        temp_theta[0,termination_boolean] = Old_pi_b[stateID,termination_boolean,option]
                    else:
                        temp_theta[0,termination_boolean] = np.clip(np.divide(np.sum(phi[option,termination_boolean,:,stateID_phi,:,0]),np.sum(phi[option,:,:,stateID_phi,:,0])),0,1)

                if match.all():
                    stateID_phi = stateID_phi + 1
                           
                temp_theta = np.divide(temp_theta, np.sum(temp_theta))
                New_pi_b[stateID,:,option] = temp_theta
        
        return New_pi_b    
        
    def Online_Baum_Welch_together(self, T_min):
        TrainingSetID = OnlineHIL.TrainingSetID(self)
        pi_hi = OnlineHIL.initialize_pi_hi(self)
        pi_b = OnlineHIL.initialize_pi_b(self)
        pi_lo = OnlineHIL.initialize_pi_lo(self)
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
        
        #Initialization 
        pi_hi_agent = PI_HI(pi_hi) 
        pi_b_agent = PI_B(pi_b)
        pi_lo_agent = PI_LO(pi_lo)
        
        zi = np.zeros((self.option_space, self.termination_space, self.option_space, 1))
        phi_h = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size,
                         self.action_space, self.termination_space, self.option_space,1))
        P_option_given_obs = np.zeros((self.option_space, 1))
        P_option_given_obs = self.mu.reshape((self.option_space, 1)) 
        phi = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space, 1))

                                        
        for t in range(0,len(self.TrainingSet)):
            
            if t==0:
                eta=1
            else:
                eta=1/(t+1) 
        
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            zi_temp1 = np.ones((self.option_space, self.termination_space, self.option_space, 1))
            phi_h_temp = np.ones((self.option_space, self.termination_space, self.option_space, StateSpace_size,  self.action_space, 
                                  self.termination_space, self.option_space, 1))
            norm = np.zeros((len(self.mu)))
            P_option_given_obs_temp = np.zeros((self.option_space, 1))
            prod_term = np.ones((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space, 
                                 self.termination_space, self.option_space))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            for ot_past in range(self.option_space):
                for bt in range(self.termination_space):
                    for ot in range(self.option_space):
                        zi_temp1[ot_past,bt,ot,0] = OnlineHIL.Pi_combined(ot, ot_past, int(Action), bt, pi_hi_agent.Policy, 
                                                                          pi_lo_agent.Policy,  pi_b_agent.Policy, int(State), self.zeta, 
                                                                          self.option_space)
                
                norm[ot_past] = P_option_given_obs[ot_past,0]*np.sum(zi_temp1[:,:,:,0],(1,2))[ot_past]
    
            zi_temp1[:,:,:,0] = np.divide(zi_temp1[:,:,:,0],np.sum(norm[:]))
            P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,0],1))*P_option_given_obs[:,0]),0) 
            
            zi = zi_temp1
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for bT in range(self.termination_space):
                                    for oT in range(self.option_space):
                                        prod_term[ot_past, bt, ot, st, at, bT, oT] = np.sum(zi[:,bT,oT,0]*np.sum(phi_h[ot_past,bt,ot,st,at,:,:,0],0))
                                        if at == int(Action) and st == int(State):
                                            phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (eta)*zi[ot_past,bt,ot,0]*P_option_given_obs[ot_past,0] 
                                            + (1-eta)*prod_term[ot_past,bt,ot,st,at,bT,oT]
                                        else:
                                            phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1-eta)*prod_term[ot_past,bt,ot,st,at,bT,oT]
                                    
            phi_h = phi_h_temp
            P_option_given_obs = P_option_given_obs_temp
            phi = np.sum(phi_h, (5,6))
            
            #M-step 
            if t > T_min:
                pi_hi = OnlineHIL.UpdatePiHi(self, pi_hi_agent.pi_hi, phi)
                pi_lo = OnlineHIL.UpdatePiLo(self, pi_lo_agent.pi_lo, phi)
                pi_b = OnlineHIL.UpdatePiB(self, pi_b_agent.pi_b, phi)
                
                pi_hi_agent = PI_HI(pi_hi) 
                pi_b_agent = PI_B(pi_b)
                pi_lo_agent = PI_LO(pi_lo)
                
        return pi_hi, pi_lo, pi_b, phi
                
                
    def Online_Baum_Welch(self, T_min):
        TrainingSetID = OnlineHIL.TrainingSetID(self)
        pi_hi = OnlineHIL.initialize_pi_hi(self)
        pi_b = OnlineHIL.initialize_pi_b(self)
        pi_lo = OnlineHIL.initialize_pi_lo(self)
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
        
        #Initialization 
        pi_hi_agent = PI_HI(pi_hi) 
        pi_b_agent = PI_B(pi_b)
        pi_lo_agent = PI_LO(pi_lo)
        
        rho = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space, self.option_space, 1)) #rho filter initialiazation
        chi = np.zeros((self.option_space, 1)) #chi filter
        chi = self.mu.reshape((self.option_space, 1)) #chi filter initialization
        phi = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, 
                        self.action_space, 1))
        

        for t in range(0,len(self.TrainingSet)):
            
            if t==0:
                eta=0.5
            else:
                eta=0.5 
        
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            chi_temp_partial = np.zeros((self.option_space, self.termination_space, self.option_space, 1)) #store partial value of chi
            norm_chi = np.zeros((len(self.mu))) #store normalizing factor for chi
            chi_temp = np.zeros((self.option_space, 1)) #store final chi value temporary
            r_temp_partial = np.zeros((self.option_space, self.termination_space, self.option_space, 1)) #r numerator
            norm_r = np.zeros((len(self.mu),len(self.mu))) #normilizing factor for r
            rho_temp = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space,  
                                self.option_space, 1))
            prod_term = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space, 
                                 self.option_space, self.option_space))
            phi_temp = np.zeros((self.option_space, self.termination_space, self.option_space, StateSpace_size, self.action_space,  
                                self.option_space, 1))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            for oT_past in range(self.option_space):
                for oT in range(self.option_space):
                    for bT in range(self.termination_space):
                        chi_temp_partial[oT_past,bT,oT,0] = OnlineHIL.Pi_combined(oT, oT_past, int(Action), bT, pi_hi_agent.Policy, 
                                                                                  pi_lo_agent.Policy,  pi_b_agent.Policy, int(State), self.zeta, 
                                                                                  self.option_space)
                        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(bT, oT, oT_past, pi_hi_agent.Policy, int(State), self.zeta, self.option_space),0.0001,1)
                        Pi_b_eval = np.clip(OnlineHIL.Pi_b(bT, pi_b_agent.Policy, int(State), oT_past),0.0001,1)
                        r_temp_partial[oT_past,bT,oT,0] = Pi_hi_eval*Pi_b_eval*chi[oT_past,0]
                        
                    norm_r[oT_past,oT] = chi[oT_past,0]*np.sum(r_temp_partial[:,:,:,0],(1))[oT_past,oT]
                norm_chi[oT_past] = chi[oT_past,0]*np.sum(chi_temp_partial[:,:,:,0],(1,2))[oT_past]
    
            chi_temp_partial[:,:,:,0] = np.divide(chi_temp_partial[:,:,:,0],np.sum(norm_chi[:]))
            chi_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(chi_temp_partial[:,:,:,0],1))*chi[:,0]),0) #next step chi
            norm_r = np.sum(norm_r,0)
            
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for oT in range(self.option_space):
                                    for oT_past in range(self.option_space):
                                        prod_term[ot_past, bt, ot, st, at, oT, oT_past] = rho[ot_past,bt,ot,st,at,oT_past,0]*np.sum(np.divide(r_temp_partial[oT_past,:,oT],norm_r[oT]))
                                        
                                    if at == int(Action) and st == int(State):
                                        rho_temp[ot_past,bt,ot,st,at,oT,0] = (eta)*np.divide(r_temp_partial[ot_past,bt,ot],norm_r[ot]) 
                                        + (1-eta)*np.sum(prod_term[ot_past,bt,ot,st,at,oT,:])
                                        phi_temp[ot_past,bt,ot,st,at,oT,0] = rho_temp[ot_past,bt,ot,st,at,oT,0]*chi_temp[oT,0] 
                                    else:
                                        rho_temp[ot_past,bt,ot,st,at,oT,0] = (1-eta)*np.sum(prod_term[ot_past,bt,ot,st,at,oT,:])
                                        phi_temp[ot_past,bt,ot,st,at,oT,0] = rho_temp[ot_past,bt,ot,st,at,oT,0]*chi_temp[oT,0] 
                                        
            chi = chi_temp
            rho = rho_temp
            phi = np.sum(phi_temp,5)
            
            #M-step 
            if t > T_min:
                pi_hi = OnlineHIL.UpdatePiHi(self, pi_hi_agent.pi_hi, phi)
                pi_lo = OnlineHIL.UpdatePiLo(self, pi_lo_agent.pi_lo, phi)
                pi_b = OnlineHIL.UpdatePiB(self, pi_b_agent.pi_b, phi)
                
                pi_hi_agent = PI_HI(pi_hi) 
                pi_b_agent = PI_B(pi_b)
                pi_lo_agent = PI_LO(pi_lo)
                
        return pi_hi, pi_lo, pi_b, chi, rho, phi              
                                        
                                        
                                        
                                        
                                        
                                        