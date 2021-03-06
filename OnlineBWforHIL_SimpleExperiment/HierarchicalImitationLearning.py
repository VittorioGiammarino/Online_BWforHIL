#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:57:36 2020

@author: vittorio
"""

import tensorflow as tf 
import numpy as np
# import argparse
# import os
# import Simulation as sim
from tensorflow import keras
import tensorflow.keras.backend as kb
# import BehavioralCloning as bc
import concurrent.futures
import csv

def match_vectors(vector1,vector2):
    
    result = np.empty((0),int)
    
    for i in range(len(vector1)):
        for j in range(len(vector2)):
            if vector1[i]==vector2[j]:
                result = np.append(result, int(vector1[i]))
                
    return result
                

def NN_options(option_space,size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input,)),
    keras.layers.Dense(option_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_options.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_actions(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input+1,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_actions.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_termination(termination_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input+1,)),
    keras.layers.Dense(termination_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_termination.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

class pi_hi_discrete:
    def __init__(self, theta_1, theta_2):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.P = np.array([[theta_1, 1-theta_1],[1-theta_2, theta_2]])
        
    def policy(self, state):
        prob_hi = self.P[int(state),:]
        prob_hi = prob_hi.reshape(1,len(prob_hi))
        
        return prob_hi
    
class pi_lo_discrete:
    def __init__(self, theta_1, theta_2, theta_3, theta_4):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.theta_4 = theta_4
        P = np.array([[theta_1, 1-theta_1], [1-theta_2, theta_2], [theta_3, 1-theta_3], [1-theta_4, theta_4]])
        self.P = P.reshape((2,2,2))
        
    def policy(self,state_and_option):
        state = state_and_option[0,0]
        option = state_and_option[0,1]
        prob_lo = self.P[int(state),int(option),:]
        prob_lo = prob_lo.reshape(1,len(prob_lo))
        
        return prob_lo
    
class pi_b_discrete:
    def __init__(self, theta_1, theta_2, theta_3, theta_4):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.theta_4 = theta_4
        P = np.array([[theta_1, 1-theta_1], [1-theta_2, theta_2], [theta_3, 1-theta_3], [1-theta_4, theta_4]])
        self.P = P.reshape((2,2,2))
        
    def policy(self,state_and_option):
        state = state_and_option[0,0]
        option = state_and_option[0,1]
        prob_b = self.P[int(state),int(option),:]
        prob_b = prob_b.reshape(1,len(prob_b))
        
        return prob_b
    
def get_discrete_policy(Theta):
    pi_hi = pi_hi_discrete(Theta[0], Theta[1])
    pi_lo = pi_lo_discrete(Theta[2], Theta[3], Theta[4], Theta[5])
    pi_b = pi_b_discrete(Theta[6], Theta[7], Theta[8], Theta[9])
    
    return pi_hi, pi_lo, pi_b

def Pi_hi(ot, Pi_hi_parameterization, state):

    Pi_hi = Pi_hi_parameterization(state)
    o_prob = Pi_hi[0,ot]
    
    return o_prob

def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
    if b == True:
        o_prob_tilde = Pi_hi(ot, Pi_hi_parameterization, state)
    elif ot == ot_past:
        o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
    else:
        o_prob_tilde = np.divide(zeta,option_space)
        
    return o_prob_tilde

def Pi_lo(a, Pi_lo_parameterization, state_and_option):
    Pi_lo = Pi_lo_parameterization(state_and_option)
    a_prob = Pi_lo[0,int(a)]
    
    return a_prob

def Pi_b(b, Pi_b_parameterization, state_and_option):
    Pi_b = Pi_b_parameterization(state_and_option)
    if b == True:
        b_prob = Pi_b[0,1]
    else:
        b_prob = Pi_b[0,0]
        
    return b_prob
    
def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
    Pi_hi_eval = np.clip(Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
    Pi_lo_eval = np.clip(Pi_lo(a, Pi_lo_parameterization, np.append(state, [[ot]],axis=1)),0.0001,1)
    Pi_b_eval = np.clip(Pi_b(b, Pi_b_parameterization, np.append(state, [[ot_past]],axis=1)),0.0001,1)
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
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
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
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
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
                    beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*Pi_combined(ot_next, ot, a, b_next, 
                                                                                       Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                       Pi_b_parameterization, state, zeta, option_space)
    beta = np.divide(beta,np.sum(beta))
    
    return beta

def Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination):
    alpha = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(len(TrainingSet)):
        print('alpha iter', t+1, '/', len(TrainingSet))
        if t ==0:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardFirstRecursion(mu, action, NN_options, 
                                                 NN_actions, NN_termination, 
                                                 state, zeta, option_space, termination_space)
        else:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardRecursion(alpha[:,:,t-1], action, NN_options, 
                                            NN_actions, NN_termination, 
                                            state, zeta, option_space, termination_space)
           
    return alpha

def Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination):
    beta = np.empty((option_space,termination_space,len(TrainingSet)))
    beta[:,:,len(TrainingSet)-1] = np.divide(np.ones((option_space,termination_space)),2*option_space)
    
    for t_raw in range(len(TrainingSet)-1):
        t = len(TrainingSet) - (t_raw+1)
        print('beta iter', t_raw+1, '/', len(TrainingSet)-1)
        state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
        action = labels[t]
        beta[:,:,t-1] = BackwardRecursion(beta[:,:,t], action, NN_options, 
                                        NN_actions, NN_termination, state, zeta, 
                                        option_space, termination_space)
        
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
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*Pi_combined(ot, ot_past, a, b, 
                                                                                  Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                  Pi_b_parameterization, state, zeta, option_space)
            gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
    gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
    return gamma_tilde

def Gamma(TrainingSet, option_space, termination_space, alpha, beta):
    gamma = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(len(TrainingSet)):
        print('gamma iter', t+1, '/', len(TrainingSet))
        gamma[:,:,t]=Smoothing(option_space, termination_space, alpha[:,:,t], beta[:,:,t])
        
    return gamma

def GammaTilde(TrainingSet, labels, beta, alpha, Pi_hi_parameterization, Pi_lo_parameterization, 
               Pi_b_parameterization, zeta, option_space, termination_space):
    gamma_tilde = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(1,len(TrainingSet)):
        print('gamma tilde iter', t, '/', len(TrainingSet)-1)
        state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
        action = labels[t]
        gamma_tilde[:,:,t]=DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                           Pi_hi_parameterization, Pi_lo_parameterization, 
                                           Pi_b_parameterization, state, zeta, option_space, termination_space)
        
    return gamma_tilde
    
def TrainingSetTermination(TrainingSet,option_space, size_input):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_reshaped_termination = np.empty((int(option_space*(T-1)),size_input+1))
    j=1
    for i in range(0,option_space*(T-1),option_space):
        for k in range(option_space):
            TrainingSet_reshaped_termination[i+k,:] = np.append(TrainingSet[j,:], [[k]])
        j+=1
        
    return TrainingSet_reshaped_termination
        
def GammaTildeReshape(gamma_tilde, option_space):
    T = gamma_tilde.shape[2]
    gamma_tilde_reshaped = np.empty((int(option_space*(T-1)),2),dtype='float32')
    j=1
    for i in range(0,option_space*(T-1),option_space):
        gamma_tilde_reshaped[i:i+option_space,:] = gamma_tilde[:,:,j]
        j+=1
        
    return gamma_tilde_reshaped

def OptimizeNNtermination(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, T, optimizer):
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_termination.trainable_weights)
            pi_b = NN_termination(TrainingSetTermination,training=True)
            loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_termination, NN_termination.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_termination.trainable_weights))
        print('termination loss:', float(loss_termination))
        
    return loss_termination
    
        
def TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels, size_input):
    TrainingSet_reshaped_actions = np.empty((int(option_space*(T)),size_input+1))
    labels_reshaped = np.empty((int(option_space*(T)),1))
    j=0
    for i in range(0,option_space*(T),option_space):
        for k in range(option_space):
            TrainingSet_reshaped_actions[i+k,:] = np.append(TrainingSet[j,:], [[k]])
            labels_reshaped[i+k,:] = labels[j]
        j+=1
        
    return TrainingSet_reshaped_actions, labels_reshaped

def GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped):
    gamma_reshaped = np.empty((int(option_space*(T)),2),dtype='float32')
    j=0
    for i in range(0,option_space*(T),option_space):
        gamma_reshaped[i:i+option_space,:] = gamma[:,:,j]
        j+=1
    
    gamma_actions_false = np.empty((int(option_space*T),action_space))
    for i in range(option_space*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_false[i,j]=gamma_reshaped[i,0]
            else:
                gamma_actions_false[i,j] = 0
            
    gamma_actions_true = np.empty((int(option_space*T),action_space))
    for i in range(option_space*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_true[i,j]=gamma_reshaped[i,1]
            else:
                gamma_actions_true[i,j] = 0   
                
    return gamma_actions_false, gamma_actions_true

def OptimizeNNactions(epochs, TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_actions.trainable_weights)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_action, NN_actions.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_actions.trainable_weights))
        print('action loss:', float(loss_action))
        
    return loss_action
        
def GammaReshapeOptions(T, option_space, gamma):
    gamma_reshaped_options = np.empty((T,option_space),dtype='float32')
    for i in range(T):
        gamma_reshaped_options[i,:] = gamma[:,1,i]
        
    return gamma_reshaped_options

def OptimizeNNoptions(epochs, TrainingSet, NN_options, gamma_reshaped_options, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_options.trainable_weights)
            pi_hi = NN_options(TrainingSet,training=True)
            loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_options, NN_options.trainable_weights)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_options.trainable_weights))
        print('options loss:', float(loss_options))
        
    return loss_options

def TrainingSetPiLo(TrainingSet,o, size_input):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_PiLo = np.empty((T,size_input+1))
    for i in range(T):
        TrainingSet_PiLo[i,:] = np.append(TrainingSet[i,:], [[o]])
        
    return TrainingSet_PiLo    


def RegularizedLoss1(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    pi_b, pi_hi, pi_lo, responsibilities, lambdas, T):
    
    values = -kb.sum(lambdas*responsibilities)
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action+values
    
    return loss
    

def OptimizeLossAndRegularizer1(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, lambdas, T, optimizer, option_space):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights, lambdas]
            tape.watch(weights)
            for i in range(option_space):
                ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i))),1)/T,0))
            responsibilities = ta.stack()
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            loss = RegularizedLoss1(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                                       pi_b, pi_hi, pi_lo, responsibilities, lambdas, T)
            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        optimizer.apply_gradients([(grads[3][:],lambdas)])
        print('options loss:', float(loss))
        
    return loss

            
def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    pi_b, pi_hi, pi_lo,T):
    
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action
    
    return loss    

    
def OptimizeLoss(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                 TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                 TrainingSet, NN_options, gamma_reshaped_options, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            loss = Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                        pi_b, pi_hi, pi_lo, T)
            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))
        
    return loss
    
def RegularizedLoss2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels):
    
    pi_b = NN_termination(TrainingSetTermination,training=True)
    pi_lo = NN_actions(TrainingSetActions,training=True)
    pi_hi = NN_options(TrainingSet,training=True)
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action+eta*regular_loss
    
    return loss
    

def OptimizeLossAndRegularizer2(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, T, optimizer, 
                               gamma, option_space, labels):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights, eta]
            tape.watch(weights)
            loss = RegularizedLoss2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels)

            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        optimizer.apply_gradients([(grads[3][:],eta)])
        print('options loss:', float(loss))
        
    return loss


def RegularizedLossTot(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels, size_input):
    
    pi_b = NN_termination(TrainingSetTermination,training=True)
    pi_lo = NN_actions(TrainingSetActions,training=True)
    pi_hi = NN_options(TrainingSet,training=True)
    
    # Regularization 1
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i,size_input))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
    # Regularization 2
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(option_space):
        ta = ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))),1)/T,0))
    responsibilities = ta.stack()
    
    values = kb.sum(lambdas*responsibilities) 
    loss_termination = kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = (kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    
    loss = -loss_termination - loss_options - loss_action + eta*regular_loss - values
    
    return loss



def OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                               gamma, option_space, labels, size_input):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            loss = RegularizedLossTot(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels, size_input)

            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))
        
    return loss

def OptimizeLossAndRegularizerTotBatch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                               gamma, option_space, labels, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = RegularizedLossTot(gamma_tilde_reshaped[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          gamma_reshaped_options[n*size_batch:(n+1)*size_batch,:], 
                                          gamma_actions_true[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          gamma_actions_false[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          NN_termination, NN_options, NN_actions,
                                          TrainingSetTermination[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          TrainingSetActions[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          TrainingSet[n*size_batch:(n+1)*size_batch,:], 
                                          eta, lambdas, gamma[:,:,n*size_batch:(n+1)*size_batch], 
                                          size_batch, option_space, labels, size_input)

            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
            print('options loss:', float(loss))
            
        
    return loss


def BaumWelch(EV, lambdas, eta):
    NN_Options = NN_options(EV.option_space, EV.size_input)
    NN_Actions = NN_actions(EV.action_space, EV.size_input)
    NN_Termination = NN_termination(EV.termination_space, EV.size_input)
    
    NN_Options.set_weights(EV.Triple_init.options_weights)
    NN_Actions.set_weights(EV.Triple_init.actions_weights)
    NN_Termination.set_weights(EV.Triple_init.termination_weights)
        
    T = EV.TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(EV.TrainingSet, EV.option_space, EV.size_input)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(EV.option_space,T, EV.TrainingSet, EV.labels, EV.size_input)

    for n in range(EV.N):
        print('iter', n+1, '/', EV.N)
        
        alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                      EV.zeta, NN_Options, NN_Actions, NN_Termination)
        beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                    NN_Options, NN_Actions, NN_Termination)
        gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
                                 NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)
    
        # MultiThreading Running
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f1 = executor.submit(Alpha, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
        #                          EV.zeta, NN_Options, NN_Actions, NN_Termination)
        #     f2 = executor.submit(Beta, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
        #                          NN_Options, NN_Actions, NN_Termination)  
        #     alpha = f1.result()
        #     beta = f2.result()
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f3 = executor.submit(Gamma, EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        #     f4 = executor.submit(GammaTilde, EV.TrainingSet, EV.labels, beta, alpha, 
        #                       NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)  
        #     gamma = f3.result()
        #     gamma_tilde = f4.result()
        
        print('Expectation done')
        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 10 #number of iterations for the maximization step
            
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, EV.option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, EV.option_space, EV.action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, EV.option_space, gamma)
    
    
        # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
        #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
        #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
        #                                          gamma, option_space, labels, size_input)
    
        loss = OptimizeLossAndRegularizerTotBatch(epochs, TrainingSet_Termination, NN_Termination, gamma_tilde_reshaped, 
                                                  TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true,
                                                  EV.TrainingSet, NN_Options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                                  gamma, EV.option_space, EV.labels, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options


def ValidationBW_reward(i, Experiment_Vars):
        lambdas = tf.Variable(initial_value=Experiment_Vars.gain_lambdas[i]*tf.ones((Experiment_Vars.option_space,)), trainable=False)
        eta = tf.Variable(initial_value=Experiment_Vars.gain_eta[i], trainable=False)
        NN_Termination, NN_Actions, NN_Options = BaumWelch(Experiment_Vars, lambdas, eta)
        list_triple = Triple(NN_Options, NN_Actions, NN_Termination)
        [trajHIL, controlHIL, optionHIL, 
         terminationHIL, flagHIL] = sim.HierarchicalPolicySim(Experiment_Vars.env, list_triple, 
                                                              Experiment_Vars.zeta, Experiment_Vars.mu, Experiment_Vars.max_epoch, 
                                                              100, Experiment_Vars.option_space, Experiment_Vars.size_input)
        length_traj = np.empty((0))
        for j in range(len(trajHIL)):
            length_traj = np.append(length_traj, len(trajHIL[j][:]))
        averageHIL = np.divide(np.sum(length_traj),len(length_traj))
        success_percentageHIL = np.divide(np.sum(flagHIL),len(length_traj))
        
        return list_triple, averageHIL, success_percentageHIL


def ValidationBW(labels, TrainingSet, action_space, option_space, termination_space, zeta, mu, NN_Options, NN_Actions, NN_Termination):
    T = TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(TrainingSet, option_space)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)
    
    # Uncomment for sequential Running
    alpha = Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_Options, NN_Actions, NN_Termination)
    beta = Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_Options, NN_Actions, NN_Termination)
    gamma = Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = GammaTilde(TrainingSet, labels, beta, alpha, 
                             NN_Options, NN_Actions, NN_Termination, zeta, option_space, termination_space)
    
    # MultiThreading Running
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f1 = executor.submit(hil.Alpha, TrainingSet, labels, option_space, termination_space, mu, 
        #                           zeta, NN_options, NN_actions, NN_termination)
        #     f2 = executor.submit(hil.Beta, TrainingSet, labels, option_space, termination_space, zeta, 
        #                           NN_options, NN_actions, NN_termination)  
        #     alpha = f1.result()
        #     beta = f2.result()
        
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f3 = executor.submit(hil.Gamma, TrainingSet, option_space, termination_space, alpha, beta)
        #     f4 = executor.submit(hil.GammaTilde, TrainingSet, labels, beta, alpha, 
        #                           NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)  
        #     gamma = f3.result()
        #     gamma_tilde = f4.result()
        
    print('Expectation done')
    print('Starting maximization step')
        
    gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = GammaReshapeOptions(T, option_space, gamma)
    
    pi_b = NN_Termination(TrainingSet_Termination)
    pi_lo = NN_Actions(TrainingSet_Actions)
    pi_hi = NN_Options(TrainingSet)

    loss = Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, pi_b, pi_hi, pi_lo, T)

    print(float(loss))
        
    return loss

def Regularizer1(TrainingSet, option_space, size_input, NN_actions, T, lambdas):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(option_space):
        ta = ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))),1)/T,0))
    responsibilities = ta.stack()
    values = kb.sum(lambdas*responsibilities) 
    
    return -values


def OptimizeRegularizer1Batch(epochs, NN_termination, NN_actions, 
                               TrainingSet, NN_options, lambdas, optimizer, option_space, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = Regularizer1(TrainingSet[n*size_batch:(n+1)*size_batch,:], option_space, size_input, 
                                    NN_actions, size_batch, lambdas)
            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            print('options loss:', float(loss))
               
    return loss    
  
def BaumWelchRegularizer1(EV, lambdas):
    NN_Options = NN_options(EV.option_space, EV.size_input)
    NN_Actions = NN_actions(EV.action_space, EV.size_input)
    NN_Termination = NN_termination(EV.termination_space, EV.size_input)
    
    NN_Options.set_weights(EV.Triple_init.options_weights)
    NN_Actions.set_weights(EV.Triple_init.actions_weights)
    NN_Termination.set_weights(EV.Triple_init.termination_weights)
        
    for n in range(1):
        print('iter', n, '/', EV.N)

        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 100 #number of iterations for the maximization step
                
        loss = OptimizeRegularizer1Batch(epochs, NN_Termination, NN_Actions, 
                                         EV.TrainingSet, NN_Options, lambdas, optimizer, EV.option_space, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options
    
def Regularizer2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels, size_input):
    # Regularization 1
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i,size_input))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
    return eta*regular_loss
    
def OptimizeRegularizer2Batch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, T, optimizer, 
                               gamma, option_space, labels, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = Regularizer2(gamma_tilde_reshaped[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    gamma_reshaped_options[n*size_batch:(n+1)*size_batch,:], 
                                    gamma_actions_true[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    gamma_actions_false[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    NN_termination, NN_options, NN_actions,
                                    TrainingSetTermination[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    TrainingSetActions[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    TrainingSet[n*size_batch:(n+1)*size_batch,:], 
                                    eta, gamma[:,:,n*size_batch:(n+1)*size_batch], 
                                    size_batch, option_space, labels, size_input)

            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
            print('options loss:', float(loss))
            
        
    return loss    

def BaumWelchRegularizer2(EV, eta):
    NN_Options = NN_options(EV.option_space, EV.size_input)
    NN_Actions = NN_actions(EV.action_space, EV.size_input)
    NN_Termination = NN_termination(EV.termination_space, EV.size_input)
    
    NN_Options.set_weights(EV.Triple_init.options_weights)
    NN_Actions.set_weights(EV.Triple_init.actions_weights)
    NN_Termination.set_weights(EV.Triple_init.termination_weights)
        
    T = EV.TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(EV.TrainingSet, EV.option_space, EV.size_input)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(EV.option_space,T, EV.TrainingSet, EV.labels, EV.size_input)

    for n in range(2):
        print('iter', n+1, '/', 2)
        
        # alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
        #               EV.zeta, NN_Options, NN_Actions, NN_Termination)
        # beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
        #             NN_Options, NN_Actions, NN_Termination)
        # gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        # gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
        #                          NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)
    
        # MultiThreading Running
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f1 = executor.submit(Alpha, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                                  EV.zeta, NN_Options, NN_Actions, NN_Termination)
            f2 = executor.submit(Beta, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                                  NN_Options, NN_Actions, NN_Termination)  
            alpha = f1.result()
            beta = f2.result()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f3 = executor.submit(Gamma, EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
            f4 = executor.submit(GammaTilde, EV.TrainingSet, EV.labels, beta, alpha, 
                              NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)  
            gamma = f3.result()
            gamma_tilde = f4.result()
        
        print('Expectation done')
        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 50 #number of iterations for the maximization step
            
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, EV.option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, EV.option_space, EV.action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, EV.option_space, gamma)
    
    
        # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
        #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
        #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
        #                                          gamma, option_space, labels, size_input)
    
        loss = OptimizeRegularizer2Batch(epochs, TrainingSet_Termination, NN_Termination, gamma_tilde_reshaped, 
                                                  TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true,
                                                  EV.TrainingSet, NN_Options, gamma_reshaped_options, eta, T, optimizer, 
                                                  gamma, EV.option_space, EV.labels, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options


def EvaluationBW(TrainingSet, labels, nSamples, EV, lambdas, eta):
    averageBW = np.empty((0))
    success_percentageBW = np.empty((0))

    for i in range(len(nSamples)):
        EV.TrainingSet = TrainingSet[0:nSamples[i],:]
        EV.labels = labels[0:nSamples[i]]
        NN_Termination, NN_Actions, NN_Options = BaumWelch(EV, lambdas, eta)
        Trained_triple = Triple(NN_Options, NN_Actions, NN_Termination)
        Trajs=100
        [trajBW, controlBW, OptionBW, 
         TerminationBW, flagBW]=sim.HierarchicalPolicySim(EV.env, Trained_triple, EV.zeta, EV.mu, EV.max_epoch, Trajs, EV.option_space, EV.size_input)
        
        length_trajBW = np.empty((0))
        for j in range(len(trajBW)):
            length_trajBW = np.append(length_trajBW, len(trajBW[j][:]))
        averageBW = np.append(averageBW,np.divide(np.sum(length_trajBW),len(length_trajBW)))
        success_percentageBW = np.append(success_percentageBW,np.divide(np.sum(flagBW),len(length_trajBW)))
              
    return averageBW, success_percentageBW

def HMM_order_estimation(d, EV):
    
    EV.option_space = d
    
    NN_Options = NN_options(EV.option_space, EV.size_input)
    NN_Actions = NN_actions(EV.action_space, EV.size_input)
    NN_Termination = NN_termination(EV.termination_space, EV.size_input)
    
    NN_Actions.set_weights(EV.Triple_init.actions_weights)
    NN_Termination.set_weights(EV.Triple_init.termination_weights)
    
    mu = np.ones(EV.option_space)*np.divide(1,EV.option_space) #initial option probability distribution
    EV.mu = mu
        
    T = EV.TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(EV.TrainingSet, EV.option_space, EV.size_input)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(EV.option_space,T, EV.TrainingSet, EV.labels, EV.size_input)
    
    for n in range(EV.N):
        print('iter', n+1, '/', EV.N)
        
        alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                      EV.zeta, NN_Options, NN_Actions, NN_Termination)
        beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                    NN_Options, NN_Actions, NN_Termination)
        gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
                                  NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)
    
        # MultiThreading Running
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f1 = executor.submit(Alpha, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
        #                          EV.zeta, NN_Options, NN_Actions, NN_Termination)
        #     f2 = executor.submit(Beta, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
        #                          NN_Options, NN_Actions, NN_Termination)  
        #     alpha = f1.result()
        #     beta = f2.result()
    
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f3 = executor.submit(Gamma, EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        #     f4 = executor.submit(GammaTilde, EV.TrainingSet, EV.labels, beta, alpha, 
        #                          NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)  
        #     gamma = f3.result()
        #     gamma_tilde = f4.result()
        
        print('Expectation done')
        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 30 #number of iterations for the maximization step
    
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, EV.option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, EV.option_space, EV.action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, EV.option_space, gamma)
        
        loss = OptimizeLoss(epochs, TrainingSet_Termination, NN_Termination, 
                            gamma_tilde_reshaped, TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true, EV.TrainingSet, 
                            NN_Options, gamma_reshaped_options, T, optimizer)

    print(float(loss))
        
    return loss


def BaumWelch_discrete(EV):
    
    P_Options = pi_hi_discrete(EV.Triple_init.theta_hi_1, EV.Triple_init.theta_hi_2)
    P_Actions = pi_lo_discrete(EV.Triple_init.theta_lo_1, EV.Triple_init.theta_lo_2,EV.Triple_init.theta_lo_3, EV.Triple_init.theta_lo_4)
    P_Termination = pi_b_discrete(EV.Triple_init.theta_b_1, EV.Triple_init.theta_b_2, EV.Triple_init.theta_b_3, EV.Triple_init.theta_b_4)

    state_0_index = np.where(EV.TrainingSet[:,0] == 0)[0]
    state_1_index = np.where(EV.TrainingSet[:,0] == 1)[0]

    action_0_index = np.where(EV.labels[:]==0)[0]
    action_1_index = np.where(EV.labels[:]==1)[0]
    action_0_state_0_index = match_vectors(action_0_index, state_0_index)
    action_1_state_0_index = match_vectors(action_1_index, state_0_index)
    action_0_state_1_index = match_vectors(action_0_index, state_1_index)
    action_1_state_1_index = match_vectors(action_1_index, state_1_index)
    
    for n in range(EV.N):
        print('iter', n+1, '/', EV.N)
        
        alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                          EV.zeta, P_Options.policy, P_Actions.policy, P_Termination.policy)
        beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                        P_Options.policy, P_Actions.policy, P_Termination.policy)
        gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
                                 P_Options.policy, P_Actions.policy, P_Termination.policy, EV.zeta, EV.option_space, EV.termination_space)
    
        # M step
        gamma_state_0 = gamma[:,:,state_0_index]
        gamma_state_1 = gamma[:,:,state_1_index]
        theta_hi_1 = np.clip(np.divide(np.sum(gamma_state_0[0,1,:]),np.sum(gamma_state_0[:,1,:])),0,1)
        theta_hi_2 = np.clip(np.divide(np.sum(gamma_state_1[1,1,:]),np.sum(gamma_state_1[:,1,:])),0,1)
    
        theta_lo_1 = np.clip(np.divide(np.sum(gamma[0,:,action_0_state_0_index]),np.sum(gamma_state_0[0,:,:])),0,1)
        theta_lo_2 = np.clip(np.divide(np.sum(gamma[1,:,action_1_state_0_index]),np.sum(gamma_state_0[1,:,:])),0,1)
        theta_lo_3 = np.clip(np.divide(np.sum(gamma[0,:,action_0_state_1_index]),np.sum(gamma_state_1[0,:,:])),0,1)
        theta_lo_4 = np.clip(np.divide(np.sum(gamma[1,:,action_1_state_1_index]),np.sum(gamma_state_1[1,:,:])),0,1)
    
        gamma_tilde_state_0 = gamma_tilde[:,:,state_0_index]
        gamma_tilde_state_1 = gamma_tilde[:,:,state_1_index]
        theta_b_1 = np.clip(np.divide(np.sum(gamma_tilde_state_0[0,0,:]),np.sum(gamma_tilde_state_0[0,:,:])),0,1)
        theta_b_2 = np.clip(np.divide(np.sum(gamma_tilde_state_0[1,1,:]),np.sum(gamma_tilde_state_0[1,:,:])),0,1)
        theta_b_3 = np.clip(np.divide(np.sum(gamma_tilde_state_1[0,0,:]),np.sum(gamma_tilde_state_1[0,:,:])),0,1)
        theta_b_4 = np.clip(np.divide(np.sum(gamma_tilde_state_1[1,1,:]),np.sum(gamma_tilde_state_1[1,:,:])),0,1)
    
        P_Options = pi_hi_discrete(theta_hi_1, theta_hi_2)
        P_Actions = pi_lo_discrete(theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4)
        P_Termination = pi_b_discrete(theta_b_1, theta_b_2, theta_b_3, theta_b_4)

        
    return P_Termination, P_Actions, P_Options
    
def Online_BaumWelch_discrete(EV, T_min, state_space):
    
    #initialization
    P_Options = pi_hi_discrete(EV.Triple_init.theta_hi_1, EV.Triple_init.theta_hi_2)
    P_Actions = pi_lo_discrete(EV.Triple_init.theta_lo_1, EV.Triple_init.theta_lo_2,EV.Triple_init.theta_lo_3, EV.Triple_init.theta_lo_4)
    P_Termination = pi_b_discrete(EV.Triple_init.theta_b_1, EV.Triple_init.theta_b_2, EV.Triple_init.theta_b_3, EV.Triple_init.theta_b_4)

    zi = np.ones((EV.option_space, EV.termination_space, EV.option_space, EV.action_space, state_space, 1))
    phi_h = np.ones((EV.option_space, EV.termination_space, EV.option_space, EV.action_space, state_space, EV.termination_space, EV.option_space,1))
    norm = np.zeros((len(EV.mu), EV.action_space, state_space))
    P_option_given_obs = np.zeros((EV.option_space, 1))

    State = EV.TrainingSet[0].reshape(1,EV.size_input)
    Action = EV.labels[0]

    for a1 in range(EV.action_space):
        for s1 in range(state_space):
            for o0 in range(EV.option_space):
                for b1 in range(EV.termination_space):
                    for o1 in range(EV.option_space):
                        state = s1*np.ones((1,1))
                        action = a1*np.ones((1,1))
                        zi[o0,b1,o1,a1,s1,0] = Pi_combined(o1, o0, action, b1, P_Options.policy, P_Actions.policy, P_Termination.policy, 
                                                               state, EV.zeta, EV.option_space)
                                                       
                norm[o0,a1,s1]=EV.mu[o0]*np.sum(zi[:,:,:,a1,s1,0],(1,2))[o0]
            
            zi[:,:,:,a1,s1,0] = np.divide(zi[:,:,:,a1,s1,0],np.sum(norm[:,a1,s1]))
            if a1 == int(Action) and s1 == int(State):
                P_option_given_obs[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi[:,:,:,a1,s1,0],1))*EV.mu),0) 

    for a1 in range(EV.action_space):
        for s1 in range(state_space):
            for o0 in range(EV.option_space):
                for b1 in range(EV.termination_space):
                    for o1 in range(EV.option_space):
                        for bT in range(EV.termination_space):
                            for oT in range(EV.option_space):
                                if a1 == int(Action) and s1 == int(State):
                                    phi_h[o0,b1,o1,a1,s1,bT,oT,0] = zi[o0,b1,o1,a1,s1,0]*EV.mu[o0]
                                else:
                                    phi_h[o0,b1,o1,a1,s1,bT,oT,0] = 0
            
    for t in range(1,len(EV.TrainingSet)):
        
        if np.mod(t,100)==0:
            print('iter', t, '/', len(EV.TrainingSet))
    
        #E-step
        zi_temp1 = np.ones((EV.option_space, EV.termination_space, EV.option_space, EV.action_space, state_space, 1))
        phi_h_temp = np.ones((EV.option_space, EV.termination_space, EV.option_space, EV.action_space, state_space, EV.termination_space, EV.option_space, 1))
        norm = np.zeros((len(EV.mu), EV.action_space, state_space))
        P_option_given_obs_temp = np.zeros((EV.option_space, 1))
        prod_term = np.ones((EV.option_space, EV.termination_space, EV.option_space, EV.action_space, state_space, EV.termination_space, EV.option_space))
    
        State = EV.TrainingSet[t].reshape(1,EV.size_input)
        Action = EV.labels[t]
        for at in range(EV.action_space):
            for st in range(state_space):
                for ot_past in range(EV.option_space):
                    for bt in range(EV.termination_space):
                        for ot in range(EV.option_space):
                            state = st*np.ones((1,1))
                            action = at*np.ones((1,1))
                            zi_temp1[ot_past,bt,ot,at,st,0] = Pi_combined(ot, ot_past, action, bt, P_Options.policy, 
                                                                          P_Actions.policy, P_Termination.policy, state, EV.zeta, EV.option_space)
                
                    norm[ot_past,at,st] = P_option_given_obs[ot_past,t-1]*np.sum(zi_temp1[:,:,:,at,st,0],(1,2))[ot_past]
    
                zi_temp1[:,:,:,at,st,0] = np.divide(zi_temp1[:,:,:,at,st,0],np.sum(norm[:,at,st]))
                if at == int(Action) and st == int(State):
                    P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,at,st,0],1))*P_option_given_obs[:,t-1]),0) 
            
        zi = np.concatenate((zi,zi_temp1),5)
        P_option_given_obs = np.concatenate((P_option_given_obs,P_option_given_obs_temp),1)
    
        for at in range(EV.action_space):
            for st in range(state_space):
                for ot_past in range(EV.option_space):
                    for bt in range(EV.termination_space):
                        for ot in range(EV.option_space):
                            for bT in range(EV.termination_space):
                                for oT in range(EV.option_space):
                                    prod_term[ot_past, bt, ot, at, st, bT, oT] = np.sum(zi[:,bT,oT,int(Action),int(State),t]*np.sum(phi_h[ot_past,bt,ot,at,st,:,:,t-1],0))
                                    if at == int(Action) and st == int(State):
                                        phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1/t)*zi[ot_past,bt,ot,at,st,t]*P_option_given_obs[ot_past,t-1] + (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                    else:
                                        phi_h_temp[ot_past,bt,ot,st,at,bT,oT,0] = (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                    
        phi_h = np.concatenate((phi_h,phi_h_temp),7)

        #M-step    
        if t > T_min:
            theta_hi_1 = np.clip(np.divide(np.sum(phi_h[:,1,0,0,:,:,:,t]),np.sum(phi_h[:,1,:,0,:,:,:,t])),0,1)
            theta_hi_2 = np.clip(np.divide(np.sum(phi_h[:,1,1,1,:,:,:,t]),np.sum(phi_h[:,1,:,1,:,:,:,t])),0,1)
            theta_lo_1 = np.clip(np.divide(np.sum(phi_h[:,:,0,0,0,:,:,t]),np.sum(phi_h[:,:,0,0,:,:,:,t])),0,1)
            theta_lo_2 = np.clip(np.divide(np.sum(phi_h[:,:,1,0,1,:,:,t]),np.sum(phi_h[:,:,1,0,:,:,:,t])),0,1)
            theta_lo_3 = np.clip(np.divide(np.sum(phi_h[:,:,0,1,0,:,:,t]),np.sum(phi_h[:,:,0,1,:,:,:,t])),0,1)
            theta_lo_4 = np.clip(np.divide(np.sum(phi_h[:,:,1,1,1,:,:,t]),np.sum(phi_h[:,:,1,0,:,:,:,t])),0,1)
            theta_b_1 = np.clip(np.divide(np.sum(phi_h[0,0,:,0,:,:,:,t]),np.sum(phi_h[0,:,:,0,:,:,:,t])),0,1)
            theta_b_2 = np.clip(np.divide(np.sum(phi_h[1,1,:,0,:,:,:,t]),np.sum(phi_h[1,:,:,0,:,:,:,t])),0,1)
            theta_b_3 = np.clip(np.divide(np.sum(phi_h[0,0,:,1,:,:,:,t]),np.sum(phi_h[0,:,:,1,:,:,:,t])),0,1)
            theta_b_4 = np.clip(np.divide(np.sum(phi_h[1,1,:,1,:,:,:,t]),np.sum(phi_h[1,:,:,1,:,:,:,t])),0,1)
        
            P_Options = pi_hi_discrete(theta_hi_1, theta_hi_2)
            P_Actions = pi_lo_discrete(theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4)
            P_Termination = pi_b_discrete(theta_b_1, theta_b_2, theta_b_3, theta_b_4)
            
    return P_Termination, P_Actions, P_Options

    
  
class Triple:
    def __init__(self, NN_options, NN_actions, NN_termination):
        self.NN_options = NN_options
        self.NN_actions = NN_actions
        self.NN_termination = NN_termination
        self.options_weights = NN_options.get_weights()
        self.actions_weights = NN_actions.get_weights()
        self.termination_weights = NN_termination.get_weights()
        
    def save(self, lambdas, eta):
        self.NN_options.save('Triple_models/H_model_lambda_{}_eta_{}/NN_options'.format(lambdas,eta))
        self.NN_actions.save('Triple_models/H_model_lambda_{}_eta_{}/NN_actions'.format(lambdas,eta))
        self.NN_termination.save('Triple_models/H_model_lambda_{}_eta_{}/NN_termination'.format(lambdas,eta))
        
    def load(lambdas, eta):
        NN_options = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_options'.format(lambdas,eta))
        NN_actions = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_actions'.format(lambdas,eta))
        NN_termination = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_termination'.format(lambdas,eta))
        
        return NN_options, NN_actions, NN_termination
        
    
class Experiment_design:
    def __init__(self, labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init, gain_lambdas,
                  gain_eta, env, max_epoch, speed, time):
        self.labels = labels
        self.TrainingSet = TrainingSet
        self.size_input = size_input
        self.action_space = action_space
        self.option_space = option_space
        self.termination_space = termination_space
        self.N = N
        self.zeta = zeta
        self.mu = mu
        self.Triple_init = Triple_init
        self.gain_lambdas = gain_lambdas
        self.gain_eta = gain_eta
        self.env = env
        self.max_epoch = max_epoch
        self.speed = speed
        self.time = time


class Triple_discrete:
    def __init__(self, theta_hi_1, theta_hi_2, theta_lo_1, theta_lo_2, theta_lo_3, theta_lo_4, theta_b_1, theta_b_2, theta_b_3, theta_b_4):
        self.theta_hi_1 = theta_hi_1
        self.theta_hi_2 = theta_hi_2
        self.theta_lo_1 = theta_lo_1
        self.theta_lo_2 = theta_lo_2
        self.theta_lo_3 = theta_lo_3
        self.theta_lo_4 = theta_lo_4
        self.theta_b_1 = theta_b_1
        self.theta_b_2 = theta_b_2
        self.theta_b_3 = theta_b_3
        self.theta_b_4 = theta_b_4
        
class Experiment_design_discrete:
    def __init__(self, labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init, gain_lambdas,
                  gain_eta, env, max_epoch):
        self.labels = labels
        self.TrainingSet = TrainingSet
        self.size_input = size_input
        self.action_space = action_space
        self.option_space = option_space
        self.termination_space = termination_space
        self.N = N
        self.zeta = zeta
        self.mu = mu
        self.Triple_init = Triple_init
        self.gain_lambdas = gain_lambdas
        self.gain_eta = gain_eta
        self.env = env
        self.max_epoch = max_epoch
    

class HardCoded_policy:
        
    
    def pi_hi(state, water_locations, option_space):
        
        water_clusters= [[None]*1 for _ in range(4)]
        
        water_clusters[0] = water_locations[0:7,:]
        water_clusters[1] = water_locations[7:14,:]
        water_clusters[2] = water_locations[14:21,:]
        water_clusters[3] = water_locations[21:,:]
        
        if state[0,0]>0 and state[0,1]>0:
            o = 1
        elif state[0,0]>0 and state[0,1]<0:
            o = 3
        elif state[0,0]<0 and state[0,1]<0:
            o = 2
        elif state[0,0]<=0 and state[0,1]>=0:
            o = 0
            
        encoded = tf.keras.utils.to_categorical(o,option_space)
        
        return encoded
        
        
        
    def pi_lo(state, o, water_locations, action_space, tol, selected_water):
        
        water_clusters= [[None]*1 for _ in range(4)]
        
        water_clusters[0] = water_locations[0:7,:]
        water_clusters[1] = water_locations[7:14,:]
        water_clusters[2] = water_locations[14:21,:]
        water_clusters[3] = water_locations[21:,:]
                
        if np.abs(np.sum(water_clusters[o][selected_water,:]-state)) > tol:
            angle = (np.arctan2((water_clusters[o][selected_water,1]-state[0,1]),(water_clusters[o][selected_water,0]-state[0,0]))*180)/np.pi
        else:
            selected_water += 1
            if selected_water == 7:
                selected_water = 0
            angle = (np.arctan2((water_clusters[o][selected_water,1]-state[0,1]),(water_clusters[o][selected_water,0]-state[0,0]))*180)/np.pi
        
        if angle < 0:
            angle = 360 + angle
        
        action_range = 360/action_space
        # determine ganularity of the action space
        actions = np.zeros(1)
        actions_rad = np.zeros(1)
        actions_slots = (action_range/2)*np.ones(1)

        for i in range(action_space):
            step = action_range
            step_rad = np.divide((step)*np.pi,180)
            actions = np.append(actions, actions[i]+step)
            actions_rad = np.append(actions_rad, actions_rad[i]+step_rad)
            actions_slots = np.append(actions_slots, actions_slots[i]+step)
            
        index = np.amin(np.where(angle<actions_slots))
        if actions[index] == 360:
            index = 0
        encoded = tf.keras.utils.to_categorical(index,action_space)
        noise = np.abs(np.random.normal(0,0.2,(1,action_space)))
        
        encoded = encoded + noise
        encoded = encoded/np.sum(encoded)
        
        return encoded, selected_water
        
    def pi_b(state, o_old, water_locations, tol, selected_water):
        
        water_clusters= [[None]*1 for _ in range(4)]
        
        water_clusters[0] = water_locations[0:7,:]
        water_clusters[1] = water_locations[7:14,:]
        water_clusters[2] = water_locations[14:21,:]
        water_clusters[3] = water_locations[21:,:]
        
        if np.abs(np.sum(water_clusters[o_old][selected_water,:]-state))>tol:
            b=0
            encoded = tf.keras.utils.to_categorical(b,2)
        else:
            encoded = np.array([0.7, 0.3])
        
        return encoded
        
    
    