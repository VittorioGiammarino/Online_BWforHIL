#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import gym
from sklearn.preprocessing import KBinsDiscretizer
import time, math, random
from typing import Tuple


class CartPole:
    class Expert:
        # =============================================================================
        #         Credit: Richard Brooker https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1/blob/master/cartpole.ipynb
        # =============================================================================
        def __init__(self, n_bins, lower_bounds, upper_bounds, Q_table):
            self.env = gym.make('CartPole-v1')
            self.n_bins = n_bins
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            self.Q_table = Q_table
        
        def discretizer(self, cart_position, cart_velocity, angle, pole_velocity) -> Tuple[int,...]:
            """Convert continues state intro a discrete state"""
            est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
            est.fit([self.lower_bounds, self.upper_bounds ])
            return tuple(map(int,est.transform([[cart_position, cart_velocity, angle, pole_velocity]])[0]))
        
        def policy(self, state : tuple ):
            """Choosing action based on epsilon-greedy policy"""
            return np.argmax(self.Q_table[state])
        
        def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
            """Temperal diffrence for updating Q-value of state-action pair"""
            future_optimal_value = np.max(self.Q_table[new_state])
            learned_value = reward + discount_factor * future_optimal_value
            return learned_value
        
        # Adaptive learning of Learning Rate
        def learning_rate(self, n : int , min_rate=0.1 ) -> float  :
            """Decaying learning rate"""
            return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
        
        def exploration_rate(n : int, min_rate= 0.001 ) -> float :
            """Decaying exploration rate"""
            return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
        
        def Training(self, n_episodes, seed):
            np.random.seed(seed)
            self.env.seed(seed)
            self.env.action_space.seed(seed)            
            
            for e in range(n_episodes):
                
                print(e, '/', n_episodes)
    
                # Discretize state into buckets
                current_state, done = CartPole.Expert.discretizer(self,*self.env.reset()), False
    
                while done==False:
        
                    # policy action 
                    action = CartPole.Expert.policy(self, current_state) # exploit
        
                    # insert random action
                    if np.random.random() < CartPole.Expert.exploration_rate(e) : 
                        action = self.env.action_space.sample() # explore 
         
                    # increment enviroment
                    obs, reward, done, _ = self.env.step(action)
                    new_state = CartPole.Expert.discretizer(self, *obs)
        
                    # Update Q-Table
                    lr = CartPole.Expert.learning_rate(self, e)
                    learnt_value = CartPole.Expert.new_Q_value(self, reward , new_state)
                    old_value = self.Q_table[current_state][action]
                    self.Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
        
                    current_state = new_state
        
                    # Render the cartpole environment
                    #self.env.render()
                    
            return self.Q_table
            
        def Evaluation(self, Q_trained, n_episodes, max_epoch_per_traj, seed):
            self.env._max_episode_steps = max_epoch_per_traj
            self.env.seed(seed)
            Reward_array = np.empty((0))
            obs = self.env.reset()
            size_input = len(obs)
            TrainingSet = np.empty((0,size_input))
            Labels = np.empty((0))
            
            for e in range(n_episodes):
                
                print('Expert iteration: ', e+1, '/', n_episodes)
                Reward = 0
                obs = self.env.reset()
                TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
                # Discretize state into buckets
                current_state, done = CartPole.Expert.discretizer(self,*obs), False
                
                # policy action 
                action = np.argmax(Q_trained[current_state]) # exploit
                Labels = np.append(Labels, action)
                
    
                while done==False:
                    
                    # increment enviroment
                    obs, reward, done, _ = self.env.step(action)
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                    new_state = CartPole.Expert.discretizer(self, *obs)
                    Reward = Reward + reward
        
                    current_state = new_state
                    
                    # policy action 
                    action = np.argmax(Q_trained[current_state]) # exploit
                    Labels = np.append(Labels, action)
        
                    # Render the cartpole environment
                    #self.env.render()
                    
                Reward_array = np.append(Reward_array, Reward) 
                    
            return TrainingSet, Labels, Reward_array         
        
    def Plot(x, u, o, b, name_file):
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plot_action = plt.scatter(x[:,2], x[:,3], c=o, marker='x', cmap='cool');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Option1', 'Option2'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(312, sharex=ax1)
        plot_action = plt.scatter(x[:,2], x[:,3], c=u, marker='x', cmap='winter');
        cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        plot_action = plt.scatter(x[0:-1,2], x[0:-1,3], c=b, marker='x', cmap='copper');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
        plt.xlabel('Pole Angle')
        plt.ylabel('Pole Velocity')
        plt.savefig(name_file, format='eps')
        plt.show()
            
    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b):
            self.env = gym.make('CartPole-v1').env
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories, seed):
            self.env.seed(seed)
            np.random.seed(seed)            
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
                Reward = 0
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = u
                    obs, reward, done, info = self.env.step(action)
                    obs = np.round(obs,3)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state_plus1).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    

        def HILVideoSimulation(self, directory, max_epoch_per_traj):
            self.env._max_episode_steps = max_epoch_per_traj
    
            # Record the environment
            self.env = gym.wrappers.Monitor(self.env, directory, resume=True)

            for t in range(1):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                while not done: # Start with while True
                    self.env.render()
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state = obs.reshape((1,size_input))
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                        u_tot = np.append(u_tot,u)
            
                        # given action, draw next state
                        action = u
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,3)
                        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                        if done == True:
                            u_tot = np.append(u_tot,0.5)
                            break
            
                        # Select Termination
                        # Termination
                        state_plus1 = obs.reshape((1,size_input))
                        prob_b = self.pi_b[o](state_plus1).numpy()
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi(state_plus1).numpy()
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
                    
            self.env.close()
            return x, u_tot, o_tot, b_tot           

        
    
    
class MountainCar:
    class Expert:
        def AverageExpert(TrainingSet):
            trajs = 0
            for i in range(1,len(TrainingSet)):
                if TrainingSet[i,1]==0 and (TrainingSet[i,0]>=-0.6 and TrainingSet[i,0]<=-0.4) and TrainingSet[i-1,1]!=0:
                    trajs +=1
            average = len(TrainingSet)/trajs
    
            return average

    
    def Plot(x, u, o, b, name_file):
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plot_action = plt.scatter(x[:,0], x[:,1], c=o, marker='x', cmap='cool');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Option1', 'Option2'])
        #plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(312, sharex=ax1)
        plot_action = plt.scatter(x[:,0], x[:,1], c=u, marker='x', cmap='winter');
        cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
        #plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        plot_action = plt.scatter(x[0:-1,0], x[0:-1,1], c=b, marker='x', cmap='copper');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.savefig(name_file, format='eps')
        plt.show()
        
    class Animation:        
        def MakeAnimation(x, o, u, b, name_file):
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=3800)            
            
            fig = plt.figure()
            ax1 = plt.subplot(311)
            ticks = [0, 1]
            plot_action = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='cool');
            cbar = fig.colorbar(plot_action)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(['Option1', 'Option2'])
            #plt.xlabel('Position')
            plt.ylabel('Velocity')
            ax1.set_xlim(-1.25, 0.55)
            ax1.set_ylim(-0.07, 0.07)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2 = plt.subplot(312, sharex=ax1)
            plot_option = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='winter');
            cbar = fig.colorbar(plot_option, ticks=[0, 0.5, 1])
            cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
            #plt.xlabel('Position')
            plt.ylabel('Velocity')
            ax2.set_ylim(-0.07, 0.07)
            plt.setp(ax2.get_xticklabels(), visible=False)
            ax3 = plt.subplot(313, sharex=ax1)
            plot_termination = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='copper');
            cbar = fig.colorbar(plot_termination, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            ax3.set_ylim(-0.07, 0.07)
            plt.show()            
            
            def animation_frame(i, x, o, u, b):           
                plot_action.set_offsets(x[0:i,:])
                plot_action.set_sizes(10*np.ones(i))
                plot_action.set_array(o[0:i])
                plot_option.set_offsets(x[0:i,:])
                plot_option.set_sizes(10*np.ones(i))
                plot_option.set_array(u[0:i])
                plot_termination.set_offsets(x[0:i,:])
                plot_termination.set_sizes(10*np.ones(i))
                plot_termination.set_array(b[0:i])
                return plot_action, plot_option, plot_termination

            animation = anim.FuncAnimation(fig, func = animation_frame, frames=b.shape[0], fargs=(x, o, u, b))
            animation.save(name_file, writer=writer)
    
    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b):
            self.env = gym.make('MountainCar-v0').env
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            flag = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = u*2
                    obs, reward, done, info = self.env.step(action)
                    obs = np.round(obs,3)
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state_plus1).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                flag = np.append(flag,done)
        
            return traj, control, Option, Termination, flag      

        def HILVideoSimulation(self, directory, max_epoch_per_traj):
            self.env._max_episode_steps = max_epoch_per_traj
    
            # Record the environment
            self.env = gym.wrappers.Monitor(self.env, directory, resume=True)

            for t in range(1):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                while not done: # Start with while True
                    self.env.render()
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state = obs.reshape((1,size_input))
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                        u_tot = np.append(u_tot,u)
            
                        # given action, draw next state
                        action = u*2
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,3)
                        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                        if done == True:
                            u_tot = np.append(u_tot,0.5)
                            break
            
                        # Select Termination
                        # Termination
                        state_plus1 = obs.reshape((1,size_input))
                        prob_b = self.pi_b[o](state_plus1).numpy()
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi(state_plus1).numpy()
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
                    
            self.env.close()
            return x, u_tot, o_tot, b_tot                
                 
            
            
            
            
            
            
            
            
            
            