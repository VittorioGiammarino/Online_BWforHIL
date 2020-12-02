#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:29:37 2020

@author: vittorio
"""

import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb

def ProcessData(traj,control,psi,stateSpace):
# =============================================================================
#     It takes raw data from expert's trajectories, concatenates them together 
# and return the training data set and the labels
# =============================================================================
    Xtr = np.empty((1,0),int)
    inputs = np.empty((3,0),int)

    for i in range(len(traj)):
        Xtr = np.append(Xtr, control[i][:])
        inputs = np.append(inputs, np.transpose(np.concatenate((stateSpace[traj[i][:-1],:], psi[i][:-1].reshape(len(psi[i])-1,1)),1)), axis=1) 
    
    labels = Xtr.reshape(len(Xtr),1)
    TrainingSet = np.transpose(inputs) 
    
    return labels, TrainingSet

class NN_PI_LO:
# =============================================================================
#     class for Neural network for pi_lo
# =============================================================================
    def __init__(self, action_space, size_input):
        self.action_space = action_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(30, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.action_space),
                keras.layers.Softmax()
                                 ])              
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_lo.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model           
            
class NN_PI_B:
# =============================================================================
#     class for Neural network for pi_b
# =============================================================================
    def __init__(self, termination_space, size_input):
        self.termination_space = termination_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(30, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.termination_space),
                keras.layers.Softmax()
                                 ])               
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_b.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model   
            
class NN_PI_HI:
# =============================================================================
#     class for Neural Network for pi_hi
# =============================================================================
    def __init__(self, option_space, size_input):
        self.option_space = option_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(100, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.option_space),
                keras.layers.Softmax()
                                ])                
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_hi.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)                
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model   
    
class OnlineHIL:
    def __init__(self, TrainingSet, Labels, option_space, M_step_epoch, optimizer):
        self.TrainingSet = TrainingSet
        self.Labels = Labels
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.action_space = int(np.max(Labels)+1)
        self.termination_space = 2
        self.zeta = 0.0001
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        pi_hi = NN_PI_HI(self.option_space, self.size_input)
        self.NN_options = pi_hi.NN_model()
        NN_low = []
        NN_termination = []
        pi_lo = NN_PI_LO(self.action_space, self.size_input)
        pi_b = NN_PI_B(self.termination_space, self.size_input)
        for options in range(self.option_space):
            NN_low.append(pi_lo.NN_model())
            NN_termination.append(pi_b.NN_model())
        self.NN_actions = NN_low
        self.NN_termination = NN_termination
        self.epochs = M_step_epoch
        self.optimizer = optimizer
        
    def FindStateIndex(self, value):
        stateSpace = np.unique(self.TrainingSet, axis=0)
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

    def Pi_lo(a, Pi_lo_parameterization, state):
        Pi_lo = Pi_lo_parameterization(state)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state):
        Pi_b = Pi_b_parameterization(state)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(OnlineHIL.Pi_lo(a, Pi_lo_parameterization, state),0.0001,1)
        Pi_b_eval = np.clip(OnlineHIL.Pi_b(b, Pi_b_parameterization, state),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
        
    def TrainingSetID(self):
        TrainingSetID = np.empty((0,1))
        for i in range(len(self.TrainingSet)):
            ID = OnlineHIL.FindStateIndex(self,self.TrainingSet[i,:].reshape(1,self.size_input))
            TrainingSetID = np.append(TrainingSetID, [[ID]], axis=0)
            
        return TrainingSetID
    
    def Loss(self, phi_h, NN_termination, NN_options, NN_actions):
# =============================================================================
#         compute Loss function to minimize
# =============================================================================
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
        loss = 0
        loss_pi_hi = 0
        loss_pi_b = 0
        loss_pi_lo = 0
        
        for at in range(self.action_space):
            for st in range(StateSpace_size):
                for ot_past in range(self.option_space):
                    for ot in range(self.option_space):
                        for bT in range(self.termination_space):
                            for oT in range(self.option_space):
                                state_input = stateSpace[st,:].reshape(1,self.size_input)
                                loss_pi_hi = loss_pi_hi - phi_h[ot_past,1,ot,at,st,bT,oT]*kb.log(NN_options(state_input,training=True)[0][ot])
                                for bt in range(self.termination_space):
                                    state_input = stateSpace[st,:].reshape(1,self.size_input)
                                    loss_pi_lo = loss_pi_lo - phi_h[ot_past,bt,ot,at,st,bT,oT]*kb.log(NN_actions[ot](state_input,training=True)[0][at])
                                    loss_pi_b = loss_pi_b - phi_h[ot_past,bt,ot,at,st,bT,oT]*kb.log(NN_termination[ot_past](state_input,training=True)[0][bt])
                                    
        loss = loss_pi_hi + loss_pi_lo + loss_pi_b
        return loss

    def OptimizeLoss(self, phi_h, t):
# =============================================================================
#         minimize Loss all toghether
# =============================================================================
        weights = []
        loss = 0
        T = len(self.TrainingSet)
        if t+1 == T:
            M_step_epochs = 100
        else:
            M_step_epochs = self.epochs
                
        for epoch in range(M_step_epochs):
            print('\nStart m-step for sample ', t,' iteration ', epoch+1)
        
            with tf.GradientTape() as tape:
                for i in range(self.option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = OnlineHIL.Loss(self, phi_h, self.NN_termination, self.NN_options, self.NN_actions)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*self.option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            print('options loss:', float(loss))
        
        return loss        
        
    def Online_Baum_Welch(self, T_min):
        TrainingSetID = OnlineHIL.TrainingSetID(self)
        stateSpace = np.unique(self.TrainingSet, axis=0)
        StateSpace_size = len(stateSpace)
        
        
        zi = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 1))
        phi_h = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, 
                         StateSpace_size, self.termination_space, self.option_space,1))
        norm = np.zeros((len(self.mu), self.action_space, StateSpace_size))
        P_option_given_obs = np.zeros((self.option_space, 1))

        State = TrainingSetID[0,0]
        Action = self.Labels[0]
        
        for a1 in range(self.action_space):
            for s1 in range(StateSpace_size):
                for o0 in range(self.option_space):
                    for b1 in range(self.termination_space):
                        for o1 in range(self.option_space):
                            state = stateSpace[s1,:].reshape(1,self.size_input)
                            action = a1
                            zi[o0,b1,o1,a1,s1,0] = OnlineHIL.Pi_combined(o1, o0, action, b1, self.NN_options, self.NN_actions[o1], self.NN_termination[o0], 
                                                                         state, self.zeta, self.option_space)
                                                       
                    norm[o0,a1,s1]=self.mu[o0]*np.sum(zi[:,:,:,a1,s1,0],(1,2))[o0]
            
                zi[:,:,:,a1,s1,0] = np.divide(zi[:,:,:,a1,s1,0],np.sum(norm[:,a1,s1]))
                if a1 == int(Action) and s1 == int(State):
                    P_option_given_obs[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi[:,:,:,a1,s1,0],1))*self.mu),0) 

        for a1 in range(self.action_space):
            for s1 in range(StateSpace_size):
                for o0 in range(self.option_space):
                    for b1 in range(self.termination_space):
                        for o1 in range(self.option_space):
                            for bT in range(self.termination_space):
                                for oT in range(self.option_space):
                                    if a1 == int(Action) and s1 == int(State):
                                        phi_h[o0,b1,o1,a1,s1,bT,oT,0] = zi[o0,b1,o1,a1,s1,0]*self.mu[o0]
                                    else:
                                        phi_h[o0,b1,o1,a1,s1,bT,oT,0] = 0
                                        
        for t in range(1,len(self.TrainingSet)):
        
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            zi_temp1 = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 1))
            phi_h_temp = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 
                                  self.termination_space, self.option_space, 1))
            norm = np.zeros((len(self.mu), self.action_space, StateSpace_size))
            P_option_given_obs_temp = np.zeros((self.option_space, 1))
            prod_term = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 
                                 self.termination_space, self.option_space))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                state = stateSpace[st,:].reshape(1,self.size_input)
                                action = at
                                zi_temp1[ot_past,bt,ot,at,st,0] = OnlineHIL.Pi_combined(ot, ot_past, action, bt, self.NN_options, 
                                                                                        self.NN_actions[ot],  self.NN_termination[ot_past], state, self.zeta, 
                                                                                        self.option_space)
                
                        norm[ot_past,at,st] = P_option_given_obs[ot_past,0]*np.sum(zi_temp1[:,:,:,at,st,0],(1,2))[ot_past]
    
                    zi_temp1[:,:,:,at,st,0] = np.divide(zi_temp1[:,:,:,at,st,0],np.sum(norm[:,at,st]))
                    if at == int(Action) and st == int(State):
                        P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,at,st,0],1))*P_option_given_obs[:,0]),0) 
            
            zi = zi_temp1
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for bT in range(self.termination_space):
                                    for oT in range(self.option_space):
                                        prod_term[ot_past, bt, ot, at, st, bT, oT] = np.sum(zi[:,bT,oT,int(Action),int(State),0]*np.sum(phi_h[ot_past,bt,ot,at,st,:,:,0],0))
                                        if at == int(Action) and st == int(State):
                                            phi_h_temp[ot_past,bt,ot,at,st,bT,oT,0] = (1/t)*zi[ot_past,bt,ot,at,st,0]*P_option_given_obs[ot_past,0] 
                                            + (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                        else:
                                            phi_h_temp[ot_past,bt,ot,at,st,bT,oT,0] = (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                    
            phi_h = phi_h_temp
            P_option_given_obs = P_option_given_obs_temp
            
            #M-step 
            if t > T_min:
                loss = OnlineHIL.OptimizeLoss(self, phi_h, t)
                
        print('Maximization done, Total Loss:',float(loss))
                
        return self.NN_options, self.NN_actions, self.NN_termination
                
                
                
                                        
                                        
                                        
                                        
                                        
                                        