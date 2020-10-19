#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt

class TwoRewards:
    class Environment:
        def __init__(self):
            self.Nc = 10 #Time steps required to bring drone to base when it crashes
            self.P_WIND = 0.1 #Gust of wind probability
            #IDs of elements in map
            self.FREE = 0
            self.TREE = 1
            self.SHOOTER = 2
            self.REWARD1 = 3
            self.REWARD2 = 4
            self.BASE = 5
            #Actions index
            self.NORTH = 0
            self.SOUTH = 1
            self.EAST = 2
            self.WEST = 3
            self.HOVER = 4
            
            def GenerateMap(self):
                
                mapsize = [10, 11]
                grid = np.zeros((mapsize[0], mapsize[1]))
            
                #define obstacles
                grid[0,5] = self.TREE
                grid[3:7,5]= self.TREE
                grid[mapsize[0]-1,5]= self.TREE

                #count trees
                ntrees=0;
                trees = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if grid[i,j]== self.TREE:
                            trees = np.append(trees, [[j, i]], 0)
                            ntrees += 1

                #R1
                reward1 = np.array([1, 8])
                grid[reward1[1],reward1[0]] = self.REWARD1

                #R2
                reward2 = np.array([9, 8])
                grid[reward2[1], reward2[0]] = self.REWARD2

                #base
                base = np.array([1, 1])
                grid[base[1],base[0]] = self.BASE

                plt.figure()
                plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
                plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                         [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
                plt.plot([reward1[0], reward1[0], reward1[0]+1, reward1[0]+1, reward1[0]],
                         [reward1[1], reward1[1]+1, reward1[1]+1, reward1[1], reward1[1]],'k-')
                plt.plot([reward2[0], reward2[0], reward2[0]+1, reward2[0]+1, reward2[0]],
                         [reward2[1], reward2[1]+1, reward2[1]+1, reward2[1], reward2[1]],'k-')

                for i in range(0,ntrees):
                    plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                             [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

                plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                         [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
                plt.fill([reward1[0], reward1[0], reward1[0]+1, reward1[0]+1, reward1[0]],
                         [reward1[1], reward1[1]+1, reward1[1]+1, reward1[1], reward1[1]],'y')
                plt.fill([reward2[0], reward2[0], reward2[0]+1, reward2[0]+1, reward2[0]],
                         [reward2[1], reward2[1]+1, reward2[1]+1, reward2[1], reward2[1]],'y')

                for i in range(0,ntrees):
                    plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                             [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

                plt.text(base[0]+0.5, base[1]+0.5, 'B')
                plt.text(reward1[0]+0.5, reward1[1]+0.5, 'R1')
                plt.text(reward2[0]+0.5, reward2[1]+0.5, 'R2')
                        
                return grid
            
            self.map = GenerateMap(self)
        
            def GenerateStateSpace(self):
                print('Generate State Space')

                stateSpace = np.empty((0,2),int)

                for m in range(0,self.map.shape[0]):
                    for n in range(0,self.map.shape[1]):
                        if self.map[m,n] != self.TREE:
                            stateSpace = np.append(stateSpace, [[m, n]], 0)
                        
                return stateSpace
            
            self.stateSpace = GenerateStateSpace(self)
                        
        def BaseStateIndex(self):

            K = self.stateSpace.shape[0];
    
            for i in range(0,self.map.shape[0]):
                for j in range(0,self.map.shape[1]):
                    if self.map[i,j]==self.BASE:
                        m=i
                        n=j
                        break
            
            for i in range(0,K):
                if self.stateSpace[i,0]==m and self.stateSpace[i,1]==n:
                    stateIndex = i
                    break
    
            return stateIndex

        def R2StateIndex(self):
        
            K = self.stateSpace.shape[0];
    
            for i in range(0,self.map.shape[0]):
                for j in range(0,self.map.shape[1]):
                    if self.map[i,j]==self.REWARD2:
                        m=i
                        n=j
                        break
            
            for i in range(0,K):
                if self.stateSpace[i,0]==m and self.stateSpace[i,1]==n:
                    stateIndex = i
                    break
    
            return stateIndex

        def R1StateIndex(self):
     
            K = self.stateSpace.shape[0];
    
            for i in range(0,self.map.shape[0]):
                for j in range(0,self.map.shape[1]):
                    if self.map[i,j]==self.REWARD1:
                        m=i
                        n=j
                        break
            
            for i in range(0,K):
                if self.stateSpace[i,0]==m and self.stateSpace[i,1]==n:
                    stateIndex = i
                    break
    
            return stateIndex

        def FindStateIndex(self, value):
    
            K = self.stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if self.stateSpace[k,0]==value[0] and self.stateSpace[k,1]==value[1]:
                    stateIndex = k
    
            return stateIndex
        
        def ComputeTransitionProbabilityMatrix(self):
            action_space=5
            K = self.stateSpace.shape[0]
            P = np.zeros((K,K,action_space))
            [M,N]=self.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = TwoRewards.Environment.FindStateIndex(self,array_temp)

                    if self.map[i,j] != self.TREE:

                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == self.EAST and self.map[i,j+1]!=self.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                                elif j==N-1 and u==self.EAST:
                                    comp_no=1
                            #west case
                            if j!=0:
                                if u==self.WEST and self.map[i,j-1]!=self.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                                elif j==0 and u==self.WEST:
                                    comp_no=1
                            #south case
                            if i!=0:
                                if u==self.SOUTH and self.map[i-1,j]!=self.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                                elif i==0 and u==self.SOUTH:
                                    comp_no=1
                            #north case
                            if i!=M-1:
                                if u==self.NORTH and self.map[i+1,j]!=self.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                                elif i==M-1 and u==self.NORTH:
                                    comp_no=1
                            #hover case
                            if u==self.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t = TwoRewards.Environment.FindStateIndex(self,array_temp)

                                # No wind case
                                P[k,t,u] = P[k,t,u]+(1-self.P_WIND)
                                base0 = TwoRewards.Environment.BaseStateIndex(self)

                                # case wind

                                #north wind
                                if s+1>N-1 or self.map[r,s+1]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t = TwoRewards.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #north wind no hit

                                #South Wind
                                if s-1<0 or self.map[r,s-1]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=TwoRewards.Environment.FindStateIndex(self,array_temp)                                 
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #south wind no hit

                                #East Wind
                                if r+1>M-1 or self.map[r+1,s]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=TwoRewards.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #east wind no hit

                                #West Wind
                                if r-1<0 or self.map[r-1,s]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=TwoRewards.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #west wind no hit

                            elif comp_no == 1:
                                base0=TwoRewards.Environment.BaseStateIndex(self)
                                P[k,base0,u]=1

            return P
      
    class Expert:
        def __init__(self):
            self.Environment = TwoRewards.Environment()
            self.R2_STATE_INDEX = self.Environment.R2StateIndex()
            self.R1_STATE_INDEX = self.Environment.R1StateIndex()
            self.P = self.Environment.ComputeTransitionProbabilityMatrix()
        
        def ComputeStageCostsR1(self):
            action_space=5
            K = self.Environment.stateSpace.shape[0]
            G = np.zeros((K,action_space))
            [M,N]=self.Environment.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = self.Environment.FindStateIndex(array_temp)

                    if self.Environment.map[i,j] != self.Environment.TREE:

                        if k == self.R1_STATE_INDEX:
                            dummy=0 #no cost
                        else:
                            for u in range(0,action_space):
                                comp_no=1;
                                # east case
                                if j!=N-1:
                                    if u == self.Environment.EAST and self.Environment.map[i,j+1]!=self.Environment.TREE:
                                        r=i
                                        s=j+1
                                        comp_no = 0
                                elif j==N-1 and u==self.Environment.EAST:
                                    comp_no=1

                                if u == self.Environment.EAST:
                                    if j==N-1 or self.Environment.map[i,j+1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #west case
                                if j!=0:
                                    if u==self.Environment.WEST and self.Environment.map[i,j-1]!=self.Environment.TREE:
                                        r=i
                                        s=j-1
                                        comp_no=0
                                elif j==0 and u==self.Environment.WEST:
                                    comp_no=1

                                if u==self.Environment.WEST:
                                    if j==0 or self.Environment.map[i,j-1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #south case
                                if i!=0:
                                    if u==self.Environment.SOUTH and self.Environment.map[i-1,j]!=self.Environment.TREE:
                                        r=i-1
                                        s=j
                                        comp_no=0
                                elif i==0 and u==self.Environment.SOUTH:
                                    comp_no=1

                                if u==self.Environment.SOUTH:
                                    if i==0 or self.Environment.map[i-1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #north case
                                if i!=M-1:
                                    if u==self.Environment.NORTH and self.Environment.map[i+1,j]!=self.Environment.TREE:
                                        r=i+1
                                        s=j
                                        comp_no=0
                                elif i==M-1 and u==self.Environment.NORTH:
                                    comp_no=1

                                if u==self.Environment.NORTH:
                                    if i==M-1 or self.Environment.map[i+1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #hover case
                                if u==self.Environment.HOVER:
                                    r=i
                                    s=j
                                    comp_no=0

                                if comp_no==0:
                                    array_temp = [r, s]

                                    G[k,u] = G[k,u]+(1-self.Environment.P_WIND) #no shot and no wind

                                    # case wind

                                    #north wind
                                    if s+1>N-1 or self.Environment.map[r,s+1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s+1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25


                                    #South Wind
                                    if s-1<0 or self.Environment.map[r,s-1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s-1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #south wind no hit

                                    #East Wind
                                    if r+1>M-1 or self.Environment.map[r+1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r+1, s]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #east wind no hit

                                    #West Wind
                                    if r-1<0 or self.Environment.map[r-1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r-1, s]
                                    
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #west wind no hit

                                elif comp_no == 1:
                                    dummy=0

            for l in range(0,action_space):
                G[self.R1_STATE_INDEX,l]=0

            return G            

        def ComputeStageCostsR2(self):
            action_space=5
            K = self.Environment.stateSpace.shape[0]
            G = np.zeros((K,action_space))
            [M,N]=self.Environment.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = self.Environment.FindStateIndex(array_temp)

                    if self.Environment.map[i,j] != self.Environment.TREE:

                        if k == self.R2_STATE_INDEX:
                            dummy=0 #no cost
                        else:
                            for u in range(0,action_space):
                                comp_no=1;
                                # east case
                                if j!=N-1:
                                    if u == self.Environment.EAST and self.Environment.map[i,j+1]!=self.Environment.TREE:
                                        r=i
                                        s=j+1
                                        comp_no = 0
                                elif j==N-1 and u==self.Environment.EAST:
                                    comp_no=1

                                if u == self.Environment.EAST:
                                    if j==N-1 or self.Environment.map[i,j+1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #west case
                                if j!=0:
                                    if u==self.Environment.WEST and self.Environment.map[i,j-1]!=self.Environment.TREE:
                                        r=i
                                        s=j-1
                                        comp_no=0
                                elif j==0 and u==self.Environment.WEST:
                                    comp_no=1

                                if u==self.Environment.WEST:
                                    if j==0 or self.Environment.map[i,j-1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #south case
                                if i!=0:
                                    if u==self.Environment.SOUTH and self.Environment.map[i-1,j]!=self.Environment.TREE:
                                        r=i-1
                                        s=j
                                        comp_no=0
                                elif i==0 and u==self.Environment.SOUTH:
                                    comp_no=1

                                if u==self.Environment.SOUTH:
                                    if i==0 or self.Environment.map[i-1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #north case
                                if i!=M-1:
                                    if u==self.Environment.NORTH and self.Environment.map[i+1,j]!=self.Environment.TREE:
                                        r=i+1
                                        s=j
                                        comp_no=0
                                elif i==M-1 and u==self.Environment.NORTH:
                                    comp_no=1

                                if u==self.Environment.NORTH:
                                    if i==M-1 or self.Environment.map[i+1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #hover case
                                if u==self.Environment.HOVER:
                                    r=i
                                    s=j
                                    comp_no=0

                                if comp_no==0:
                                    array_temp = [r, s]

                                    G[k,u] = G[k,u]+(1-self.Environment.P_WIND) #no shot and no wind

                                    # case wind

                                    #north wind
                                    if s+1>N-1 or self.Environment.map[r,s+1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s+1]
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25


                                    #South Wind
                                    if s-1<0 or self.Environment.map[r,s-1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s-1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #south wind no hit

                                    #East Wind
                                    if r+1>M-1 or self.Environment.map[r+1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r+1, s]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #east wind no hit

                                    #West Wind
                                    if r-1<0 or self.Environment.map[r-1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r-1, s]
                                    
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #west wind no hit

                                elif comp_no == 1:
                                    dummy=0

            for l in range(0,action_space):
                G[self.R2_STATE_INDEX,l]=0

            return G            

        def ComputeStageCostsR1andR2(self):
            action_space=5
            K = self.Environment.stateSpace.shape[0]
            G = np.zeros((K,action_space))
            [M,N]=self.Environment.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = self.Environment.FindStateIndex(array_temp)

                    if self.Environment.map[i,j] != self.Environment.TREE:

                        if k == self.R1_STATE_INDEX:
                            dummy=0 #no cost
                        elif k == self.R2_STATE_INDEX:
                            dummy=0
                        else:
                            for u in range(0,action_space):
                                comp_no=1;
                                # east case
                                if j!=N-1:
                                    if u == self.Environment.EAST and self.Environment.map[i,j+1]!=self.Environment.TREE:
                                        r=i
                                        s=j+1
                                        comp_no = 0
                                elif j==N-1 and u==self.Environment.EAST:
                                    comp_no=1

                                if u == self.Environment.EAST:
                                    if j==N-1 or self.Environment.map[i,j+1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #west case
                                if j!=0:
                                    if u==self.Environment.WEST and self.Environment.map[i,j-1]!=self.Environment.TREE:
                                        r=i
                                        s=j-1
                                        comp_no=0
                                elif j==0 and u==self.Environment.WEST:
                                    comp_no=1

                                if u==self.Environment.WEST:
                                    if j==0 or self.Environment.map[i,j-1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #south case
                                if i!=0:
                                    if u==self.Environment.SOUTH and self.Environment.map[i-1,j]!=self.Environment.TREE:
                                        r=i-1
                                        s=j
                                        comp_no=0
                                elif i==0 and u==self.Environment.SOUTH:
                                    comp_no=1

                                if u==self.Environment.SOUTH:
                                    if i==0 or self.Environment.map[i-1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #north case
                                if i!=M-1:
                                    if u==self.Environment.NORTH and self.Environment.map[i+1,j]!=self.Environment.TREE:
                                        r=i+1
                                        s=j
                                        comp_no=0
                                elif i==M-1 and u==self.Environment.NORTH:
                                    comp_no=1

                                if u==self.Environment.NORTH:
                                    if i==M-1 or self.Environment.map[i+1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #hover case
                                if u==self.Environment.HOVER:
                                    r=i
                                    s=j
                                    comp_no=0

                                if comp_no==0:
                                    array_temp = [r, s]

                                    G[k,u] = G[k,u]+(1-self.Environment.P_WIND) #no shot and no wind

                                    # case wind

                                    #north wind
                                    if s+1>N-1 or self.Environment.map[r,s+1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s+1]
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25


                                    #South Wind
                                    if s-1<0 or self.Environment.map[r,s-1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s-1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #south wind no hit

                                    #East Wind
                                    if r+1>M-1 or self.Environment.map[r+1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r+1, s]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #east wind no hit

                                    #West Wind
                                    if r-1<0 or self.Environment.map[r-1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r-1, s]
                                    
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #west wind no hit

                                elif comp_no == 1:
                                    dummy=0

            for l in range(0,action_space):
                G[self.R1_STATE_INDEX,l]=0
                G[self.R2_STATE_INDEX] = 0

            return G 
        
        def ValueIteration(self, G, TERMINAL_INDEX):
            action_space=5
            tol=10**(-5)
            K = G.shape[0]
            V=np.zeros((K,action_space))
            VV=np.zeros((K,2))
            I=np.zeros((K))
            Err=np.zeros((K))

            #initialization
            VV[:,0]=50
            VV[TERMINAL_INDEX,0]=0
            n=0
            Check_err=1

            while Check_err==1:
                n=n+1
                Check_err=0
                for k in range(0,K):
                    if n>1:
                        VV[:,0]=VV[0:,1]

                    if k==TERMINAL_INDEX:
                        VV[k,1]=0
                        V[k,:]=0
                    else:
                        CTG=np.zeros((action_space)) #cost to go
                        for u in range(0,action_space):
                            for j in range(0,K):
                                CTG[u]=CTG[u] + self.P[k,j,u]*VV[j,1]

                            V[k,u]=G[k,u]+CTG[u]

                        VV[k,1]=np.amin(V[k,:])
                        flag = np.where(V[k,:]==np.amin(V[k,:]))
                        I[k]=flag[0][0]

                    Err[k]=abs(VV[k,1]-VV[k,0])

                    if Err[k]>tol:
                        Check_err=1

            J_opt=VV[:,1]
            I[TERMINAL_INDEX]=self.Environment.HOVER
            u_opt = I[:]

            return J_opt,u_opt
        
        def ValueIterationBoth(self, G):
            
            action_space=5
            tol=10**(-5)
            K = G.shape[0]
            V=np.zeros((K,action_space))
            VV=np.zeros((K,2))
            I=np.zeros((K))
            Err=np.zeros((K))

            #initialization
            VV[:,0]=50
            VV[self.R1_STATE_INDEX,0]=0
            VV[self.R2_STATE_INDEX,0]=0
            n=0
            Check_err=1

            while Check_err==1:
                n=n+1
                Check_err=0
                for k in range(0,K):
                    if n>1:
                        VV[:,0]=VV[0:,1]

                    if k==self.R1_STATE_INDEX:
                        VV[k,1]=0
                        V[k,:]=0
                    elif k==self.R2_STATE_INDEX:
                        VV[k,1]=0
                        V[k,:]=0
                    else:
                        CTG=np.zeros((action_space)) #cost to go
                        for u in range(0,action_space):
                            for j in range(0,K):
                                CTG[u]=CTG[u] + self.P[k,j,u]*VV[j,1]

                            V[k,u]=G[k,u]+CTG[u]

                        VV[k,1]=np.amin(V[k,:])
                        flag = np.where(V[k,:]==np.amin(V[k,:]))
                        I[k]=flag[0][0]

                    Err[k]=abs(VV[k,1]-VV[k,0])

                    if Err[k]>tol:
                        Check_err=1

            J_opt=VV[:,1]
            I[self.R1_STATE_INDEX]=self.Environment.HOVER
            I[self.R2_STATE_INDEX]=self.Environment.HOVER
            u_opt = I[:]

            return J_opt,u_opt  
        
        def PlotPolicy(self, u, name):
            mapsize = self.Environment.map.shape
            #count trees
            ntrees=0;
            trees = np.empty((0,2),int)
            shooters = np.empty((0,2),int)
            nshooters=0
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.Environment.map[i,j]==self.Environment.TREE:
                        trees = np.append(trees, [[j, i]], 0)
                        ntrees += 1
                    if self.Environment.map[i,j]==self.Environment.SHOOTER:
                        shooters = np.append(shooters, [[j, i]], 0)
                        nshooters+=1

            #R1
            R1Index=self.R1_STATE_INDEX
            i_R1 = self.Environment.stateSpace[R1Index,0]
            j_R1 = self.Environment.stateSpace[R1Index,1]
            R1 = np.array([j_R1, i_R1])
            #base
            BaseIndex=self.Environment.BaseStateIndex()
            i_base = self.Environment.stateSpace[BaseIndex,0]
            j_base = self.Environment.stateSpace[BaseIndex,1]
            base = np.array([j_base, i_base])
            #R2
            R2Index = self.R2_STATE_INDEX
            i_R2 = self.Environment.stateSpace[R2Index,0]
            j_R2 = self.Environment.stateSpace[R2Index,1]
            R2 = np.array([j_R2, i_R2])

            # Plot
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
            plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
            plt.plot([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                     [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'k-')
            plt.plot([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                     [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'k-')

            for i in range(0,nshooters):
                plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

            for i in range(0,ntrees):
                plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

            plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
            plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                     [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'y')
            plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                     [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'y')

            for i in range(0,nshooters):
                plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

            for i in range(0,ntrees):
                plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

            plt.text(base[0]+0.5, base[1]+0.5, 'B')
            plt.text(R1[0]+0.5, R1[1]+0.5, 'R1')
            plt.text(R2[0]+0.5, R2[1]+0.5, 'R2')
            for i in range(0,nshooters):
                plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

            for s in range(0,u.shape[0]):
                if u[s] == self.Environment.NORTH:
                    txt = u'\u2191'
                elif u[s] == self.Environment.SOUTH:
                    txt = u'\u2193'
                elif u[s] == self.Environment.EAST:
                    txt = u'\u2192'
                elif u[s] == self.Environment.WEST:
                    txt = u'\u2190'
                elif u[s] == self.Environment.HOVER:
                    txt = u'\u2715'
                plt.text(self.Environment.stateSpace[s,1]+0.3, self.Environment.stateSpace[s,0]+0.5,txt)
            
            plt.savefig(name, format='eps')
            
        
        def ComputeFlatPolicy(self):
            GR1 = TwoRewards.Expert.ComputeStageCostsR1(self)
            GR2 = TwoRewards.Expert.ComputeStageCostsR2(self)
            GBoth = TwoRewards.Expert.ComputeStageCostsR1andR2(self)
            [JR1,UR1] = TwoRewards.Expert.ValueIteration(self,GR1,self.R1_STATE_INDEX)
            [JR2,UR2] = TwoRewards.Expert.ValueIteration(self,GR2,self.R2_STATE_INDEX)
            [JBoth,UBoth] = TwoRewards.Expert.ValueIterationBoth(self,GBoth)
            UR1 = UR1.reshape(len(UR1),1)
            UR2 = UR2.reshape(len(UR2),1)
            UBoth = UBoth.reshape(len(UBoth),1)
            UTot = np.concatenate((UR1,UR2,UBoth,self.Environment.HOVER*np.ones((len(UR1),1))),1)
            TwoRewards.Expert.PlotPolicy(self, UR1, 'Figures/FiguresExpert/Expert_R1.eps')
            TwoRewards.Expert.PlotPolicy(self, UR2, 'Figures/FiguresExpert/Expert_R2.eps')
            TwoRewards.Expert.PlotPolicy(self, UBoth, 'Figures/FiguresExpert/Expert_Both.eps')
            
            return UTot, UR1, UR2, UBoth