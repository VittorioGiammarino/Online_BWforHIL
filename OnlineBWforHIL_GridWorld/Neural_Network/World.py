#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
                        
                return grid
            
            self.map = GenerateMap(self)
        
            def GenerateStateSpace(self):

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
# =============================================================================
# Compute the expert's policy using value iteration and plot the result
# =============================================================================
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
        
        def StateSpace(self):
            stateSpace = np.empty((0,3),int)

            for m in range(0,self.Environment.map.shape[0]):
                for n in range(0,self.Environment.map.shape[1]):
                    for k in range(4):
                        if self.Environment.map[m,n] != self.Environment.TREE:
                            stateSpace = np.append(stateSpace, [[m, n, k]], 0)
                        
            return stateSpace
        
        def FindStateIndex(self, value):
            
            stateSpace = TwoRewards.Expert.StateSpace(self)
            K = stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if stateSpace[k,0]==value[0,0] and stateSpace[k,1]==value[0,1] and stateSpace[k,2]==value[0,2]:
                    stateIndex = k
    
            return stateIndex
        
        def generate_pi_hi(self):
            stateSpace = TwoRewards.Expert.StateSpace(self)
            pi_hi = np.empty((0,1),int)
            for i in range(len(stateSpace)):
                if stateSpace[i,1]<5:
                    pi_hi = np.append(pi_hi, [[0]], 0)
                elif stateSpace[i,1]>5:
                    pi_hi = np.append(pi_hi, [[1]], 0)
                elif stateSpace[i,2] == 0:
                    pi_hi = np.append(pi_hi, [[0]], 0)
                else:
                    pi_hi = np.append(pi_hi, [[1]], 0)
                 
            pi_hi_encoded = np.zeros((len(pi_hi), pi_hi.max()+1))
            pi_hi_encoded[np.arange(len(pi_hi)),pi_hi[:,0]] = 1
            
            return pi_hi_encoded
        
        def generate_pi_lo(self, Uopt, Ugeneral, pi_hi, n_op):
            stateSpace = TwoRewards.Expert.StateSpace(self)
            pi_lo = np.empty((0,1),int)
            j=0

            for i in range(len(pi_hi)):
                if pi_hi[i,n_op]==1:
                    if i!=0 and np.mod(i,4)==0:
                        j = j+1
                    psi = stateSpace[i,2]
                    pi_lo = np.append(pi_lo, [[int(Uopt[j,psi])]], 0)
                else:
                    if i!=0 and np.mod(i,4)==0:
                        j = j+1
                    pi_lo = np.append(pi_lo, [[int(Ugeneral[j,0])]], 0)
                        
            pi_lo_encoded = np.zeros((len(pi_lo), pi_lo.max()+1,1))
            pi_lo_encoded[np.arange(len(pi_lo)),pi_lo[:,0],0] = 1
            
            return pi_lo_encoded
        
        def generate_pi_b(self):
            stateSpace = TwoRewards.Expert.StateSpace(self)
            pi_b = np.empty((0,1),int)
            for i in range(len(stateSpace)):
                if stateSpace[i,1]==5:
                    pi_b = np.append(pi_b, [[1]], 0)
                else:
                    pi_b = np.append(pi_b, [[0]], 0)

            pi_b_encoded = np.zeros((len(pi_b), pi_b.max()+1, 1))
            pi_b_encoded[np.arange(len(pi_b)),pi_b[:,0],0] = 1
            
            return pi_b_encoded
        
        def HierarchicalPolicy(self):
# =============================================================================
# This function generates a hierarchical policy for expert starting from 
# the solution obtained using value-iteration. The policy is arbitrarily obtained using 
# functions already defined.
# =============================================================================
            UTot, UR1, UR2, UBoth = TwoRewards.Expert.ComputeFlatPolicy(self)
            pi_hi = TwoRewards.Expert.generate_pi_hi(self)
            pi_lo1 = TwoRewards.Expert.generate_pi_lo(self, UTot, UR1, pi_hi, 0)
            pi_lo2 = TwoRewards.Expert.generate_pi_lo(self, UTot, UR2, pi_hi, 1)
            pi_lo = np.concatenate((pi_lo1, pi_lo2), 2)
            pi_b1 = TwoRewards.Expert.generate_pi_b(self)
            pi_b2 = TwoRewards.Expert.generate_pi_b(self)
            pi_b = np.concatenate((pi_b1, pi_b2), 2)
                        
            return pi_hi, pi_lo, pi_b
        
        def PlotOptions(self, pi_hi, name):
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
                
            for s in range(0,len(pi_hi)):
                if pi_hi[s]==0:
                    c = 'c'
                elif pi_hi[s]==1:
                    c = 'lime'
                elif pi_hi[s]==2:
                    c = 'y'    
                plt.fill([self.Environment.stateSpace[s,1], self.Environment.stateSpace[s,1], self.Environment.stateSpace[s,1]+0.9, 
                          self.Environment.stateSpace[s,1]+0.9, self.Environment.stateSpace[s,1]],
                         [self.Environment.stateSpace[s,0], self.Environment.stateSpace[s,0]+0.9, self.Environment.stateSpace[s,0]+0.9, 
                          self.Environment.stateSpace[s,0], self.Environment.stateSpace[s,0]],c)            
 
            
            plt.savefig(name, format='eps')
       
        def PlotHierachicalPolicy(self):
            pi_hi, pi_lo, pi_b = TwoRewards.Expert.HierarchicalPolicy(self)
            pi_hi = np.argmax(pi_hi,1)
            pi_b = np.argmax(pi_b,1)
            pi_lo = np.argmax(pi_lo,1)
            option_space = pi_lo.shape[1]
            PI_HI = np.empty((0),int)
            PI_B = np.empty((0),int)
            U1 = np.empty((0,4,1),int)
            U2 = np.empty((0,4,1),int)
            for i in range(0,len(pi_lo),4):
                PI_HI = np.append(PI_HI, pi_hi[i])
                PI_B = np.append(PI_B, pi_b[i,0])
                u1 = pi_lo[i:i+4,0].reshape(1,4,1)
                u2 = pi_lo[i:i+4,1].reshape(1,4,1)
                U1 = np.append(U1,u1,0)
                U2 = np.append(U2,u2,0)
            U = np.concatenate((U1,U2),2)
            
            for option in range(option_space):
                for psi in range(4):
                    TwoRewards.Expert.PlotPolicy(self, U[:,psi,option], 'Figures/FiguresExpert/Hierarchical/Expert_option{}_psi{}.eps'.format(option, psi))
                    
            TwoRewards.Expert.PlotOptions(self, PI_HI, 'Figures/FiguresExpert/Hierarchical/Expert_High_policy.eps')
            TwoRewards.Expert.PlotOptions(self, PI_B, 'Figures/FiguresExpert/Hierarchical/Expert_Termination_policy.eps')
            
        class Plot:
            def __init__(self, pi_hi, pi_lo, pi_b):
                self.pi_hi = pi_hi
                self.pi_lo = pi_lo
                self.pi_b = pi_b
                self.expert = TwoRewards.Expert()
                
            def PlotHierachicalPolicy(self, NameFilePI_HI, NameFilePI_LO, NameFilePI_B):
                stateSpace = self.expert.StateSpace()
                pi_hi = np.argmax(self.pi_hi(stateSpace).numpy(),1)
                option_space = len(self.pi_lo)
                psi_space = 4
                PI_HI = np.empty((0,psi_space,1),int)
                PI_B = []
                PI_LO = []
                for i in range(option_space):
                    PI_LO.append(np.empty((0,psi_space,1),int))
                for i in range(2):
                    PI_B.append(np.empty((0,psi_space,1),int))
                for i in range(0,len(pi_hi),psi_space):
                    pi_hi_temp = pi_hi[i:i+psi_space].reshape(1,psi_space,1)
                    PI_HI = np.append(PI_HI, pi_hi_temp, 0)
                    for option in range(option_space):
                        pi_b = np.argmax(self.pi_b[option](stateSpace).numpy(),1)
                        pi_b_temp = pi_b[i:i+psi_space].reshape(1,psi_space,1)
                        PI_B[option] = np.append(PI_B[option], pi_b_temp, 0)
                        pi_lo = np.argmax(self.pi_lo[option](stateSpace).numpy(),1)
                        pi_lo_temp = pi_lo[i:i+psi_space].reshape(1,psi_space,1)
                        PI_LO[option] = np.append(PI_LO[option], pi_lo_temp, 0)
            
                for option in range(option_space):
                    for psi in range(psi_space):
                        U = PI_LO[option]
                        self.expert.PlotPolicy(U[:,psi,0], NameFilePI_LO.format(option, psi))
                        B = PI_B[option]
                        self.expert.PlotOptions(B[:,psi,0], NameFilePI_B.format(option, psi))
                        
                for psi in range(psi_space):
                    self.expert.PlotOptions(PI_HI[:,psi,0], NameFilePI_HI.format(psi))
                    
        class Simulation_tabular:
            def __init__(self, pi_hi, pi_lo, pi_b):
                option_space = pi_hi.shape[1]
                self.option_space = option_space
                self.mu = np.ones(option_space)*np.divide(1,option_space)
                self.zeta = 0.0001
                self.Environment = TwoRewards.Environment()
                self.initial_state = self.Environment.BaseStateIndex()
                self.P = self.Environment.ComputeTransitionProbabilityMatrix()
                self.stateSpace = self.Environment.stateSpace
                self.R2_STATE_INDEX = self.Environment.R2StateIndex()
                self.R1_STATE_INDEX = self.Environment.R1StateIndex()
                self.pi_hi = TwoRewards.PI_HI(pi_hi)
                self.pi_lo = TwoRewards.PI_LO(pi_lo)
                self.pi_b = TwoRewards.PI_B(pi_b)
                
            def UpdateReward(self, psi, x):
                epsilon1 = 0.9
                u1 = np.random.rand()
                epsilon2 = 0.8
                u2 = np.random.rand()
    
                if psi == 0 and u2 > epsilon2:
                    psi = 2
                elif psi == 1 and u1 > epsilon1:
                    psi = 2
                elif psi == 3 and u1 > epsilon1:
                    psi = 0
                elif psi == 3 and u2 > epsilon2:
                    psi = 1
                elif psi == 3 and u1 > epsilon1 and u2 > epsilon2:
                    psi = 2
        
                if psi == 0 and x == self.R1_STATE_INDEX:
                    psi = 3
                elif psi == 1 and x == self.R2_STATE_INDEX:
                    psi = 3 
                elif psi == 2 and x == self.R1_STATE_INDEX:
                    psi = 1
                elif psi == 2 and x == self.R2_STATE_INDEX:
                    psi = 0
        
                return psi
                
            def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
                traj = [[None]*1 for _ in range(number_of_trajectories)]
                control = [[None]*1 for _ in range(number_of_trajectories)]
                Option = [[None]*1 for _ in range(number_of_trajectories)]
                Termination = [[None]*1 for _ in range(number_of_trajectories)]
                reward = np.empty((0,0),int)
                psi_evolution = [[None]*1 for _ in range(number_of_trajectories)]
    
                for t in range(0,number_of_trajectories):
        
                    x = np.empty((0,0),int)
                    x = np.append(x, self.initial_state)
                    u_tot = np.empty((0,0))
                    o_tot = np.empty((0,0),int)
                    b_tot = np.empty((0,0),int)
                    psi_tot = np.empty((0,0),int)
                    psi = 3
                    psi_tot = np.append(psi_tot, psi)
                    r=0
        
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state_partial = self.stateSpace[x[0],:].reshape(1,len(self.stateSpace[x[0],:]))
                    state = np.concatenate((state_partial,[[psi]]),1)
                    prob_b = self.pi_b.Policy(state, o)
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
                        o_prob_tilde = self.pi_hi.Policy(state)
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
                        state_partial = self.stateSpace[x[k],:].reshape(1,len(self.stateSpace[x[k],:]))
                        state = np.concatenate((state_partial,[[psi]]),1)
                        # draw action
                        prob_u = self.pi_lo.Policy(state,o)
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
                        # given action, draw next state
                        x_k_possible=np.where(self.P[x[k],:,int(u)]!=0)
                        prob = self.P[x[k],x_k_possible[0][:],int(u)]
                        prob_rescaled = np.divide(prob,np.amin(prob))
            
                        for i in range(1,prob_rescaled.shape[0]):
                            prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
                        draw=np.divide(np.random.rand(),np.amin(prob))
                        index_x_plus1=np.amin(np.where(draw<prob_rescaled))
                        x = np.append(x, x_k_possible[0][index_x_plus1])
                        u_tot = np.append(u_tot,u)
            
                        if (x[k] == self.R1_STATE_INDEX and (psi == 0 or psi==2)) or (x[k] == self.R2_STATE_INDEX and (psi == 1 or psi==2)):
                            r = r + 1 
            
                        # Randomly update the reward
                        psi = TwoRewards.Expert.Simulation_tabular.UpdateReward(self, psi, x[k])
                        psi_tot = np.append(psi_tot, psi)
            
                        # Select Termination
                        # Termination
                        state_plus1_partial = self.stateSpace[x[k+1],:].reshape(1,len(self.stateSpace[x[k+1],:]))
                        state_plus1 = np.concatenate((state_plus1_partial,[[psi]]),1)
                        prob_b = self.pi_b.Policy(state_plus1,o)
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
                            o_prob_tilde = self.pi_hi.Policy(state_plus1)
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
                    psi_evolution[t] = psi_tot                
                    reward = np.append(reward,r)

        
                return traj, control, Option, Termination, psi_evolution, reward
            
            def HILVideoSimulation(self,u,states,o,psi,name_video):
                
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
                fig = plt.figure()
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

                ims = []
                for s in range(0,len(u)):
                    if psi[s]==0:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'m')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'y')
                    if psi[s]==1:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'y')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'m')
                    if psi[s]==2:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'m')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'m')
                    if psi[s]==3:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'y')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'y')
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
                    if o[s]==0:
                        c = 'c'
                    elif o[s]==1:
                        c = 'lime'
                    elif o[s]==2:
                        c = 'y'         
                    im1 = plt.text(self.Environment.stateSpace[states[s],1]+0.3, self.Environment.stateSpace[states[s],0]+0.1, txt, fontsize=20, backgroundcolor=c)
                    ims.append([im1,im2,im3])
        
                ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                                repeat_delay=2000)
                ani.save(name_video)
                                 
                
        class Simulation_NN:
            def __init__(self, pi_hi, pi_lo, pi_b):
                option_space = len(pi_lo)
                self.option_space = option_space
                self.mu = np.ones(option_space)*np.divide(1,option_space)
                self.zeta = 0.0001
                self.Environment = TwoRewards.Environment()
                self.initial_state = self.Environment.BaseStateIndex()
                self.P = self.Environment.ComputeTransitionProbabilityMatrix()
                self.stateSpace = self.Environment.stateSpace
                self.R2_STATE_INDEX = self.Environment.R2StateIndex()
                self.R1_STATE_INDEX = self.Environment.R1StateIndex()
                self.pi_hi = pi_hi
                self.pi_lo = pi_lo
                self.pi_b = pi_b
                
            def UpdateReward(self, psi, x):
                epsilon1 = 0.9
                u1 = np.random.rand()
                epsilon2 = 0.8
                u2 = np.random.rand()
    
                if psi == 0 and u2 > epsilon2:
                    psi = 2
                elif psi == 1 and u1 > epsilon1:
                    psi = 2
                elif psi == 3 and u1 > epsilon1:
                    psi = 0
                elif psi == 3 and u2 > epsilon2:
                    psi = 1
                elif psi == 3 and u1 > epsilon1 and u2 > epsilon2:
                    psi = 2
        
                if psi == 0 and x == self.R1_STATE_INDEX:
                    psi = 3
                elif psi == 1 and x == self.R2_STATE_INDEX:
                    psi = 3 
                elif psi == 2 and x == self.R1_STATE_INDEX:
                    psi = 1
                elif psi == 2 and x == self.R2_STATE_INDEX:
                    psi = 0
        
                return psi
                
            def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
                traj = [[None]*1 for _ in range(number_of_trajectories)]
                control = [[None]*1 for _ in range(number_of_trajectories)]
                Option = [[None]*1 for _ in range(number_of_trajectories)]
                Termination = [[None]*1 for _ in range(number_of_trajectories)]
                reward = np.empty((0,0),int)
                psi_evolution = [[None]*1 for _ in range(number_of_trajectories)]
    
                for t in range(0,number_of_trajectories):
        
                    x = np.empty((0,0),int)
                    x = np.append(x, self.initial_state)
                    u_tot = np.empty((0,0))
                    o_tot = np.empty((0,0),int)
                    b_tot = np.empty((0,0),int)
                    psi_tot = np.empty((0,0),int)
                    psi = 3
                    psi_tot = np.append(psi_tot, psi)
                    r=0
        
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state_partial = self.stateSpace[x[0],:].reshape(1,len(self.stateSpace[x[0],:]))
                    state = np.concatenate((state_partial,[[psi]]),1)
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
                        state_partial = self.stateSpace[x[k],:].reshape(1,len(self.stateSpace[x[k],:]))
                        state = np.concatenate((state_partial,[[psi]]),1)
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
                        # given action, draw next state
                        x_k_possible=np.where(self.P[x[k],:,int(u)]!=0)
                        prob = self.P[x[k],x_k_possible[0][:],int(u)]
                        prob_rescaled = np.divide(prob,np.amin(prob))
            
                        for i in range(1,prob_rescaled.shape[0]):
                            prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
                        draw=np.divide(np.random.rand(),np.amin(prob))
                        index_x_plus1=np.amin(np.where(draw<prob_rescaled))
                        x = np.append(x, x_k_possible[0][index_x_plus1])
                        u_tot = np.append(u_tot,u)
            
                        if (x[k] == self.R1_STATE_INDEX and (psi == 0 or psi==2)) or (x[k] == self.R2_STATE_INDEX and (psi == 1 or psi==2)):
                            r = r + 1 
            
                        # Randomly update the reward
                        psi = TwoRewards.Expert.Simulation_NN.UpdateReward(self, psi, x[k])
                        psi_tot = np.append(psi_tot, psi)
            
                        # Select Termination
                        # Termination
                        state_plus1_partial = self.stateSpace[x[k+1],:].reshape(1,len(self.stateSpace[x[k+1],:]))
                        state_plus1 = np.concatenate((state_plus1_partial,[[psi]]),1)
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
                    psi_evolution[t] = psi_tot                
                    reward = np.append(reward,r)

                return traj, control, Option, Termination, psi_evolution, reward
            
            def HILVideoSimulation(self,u,states,o,psi,name_video):
                
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
                fig = plt.figure()
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

                ims = []
                for s in range(0,len(u)):
                    if psi[s]==0:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'m')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'y')
                    if psi[s]==1:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'y')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'m')
                    if psi[s]==2:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'m')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'m')
                    if psi[s]==3:
                        im2, = plt.fill([R1[0], R1[0], R1[0]+1, R1[0]+1, R1[0]],
                                        [R1[1], R1[1]+1, R1[1]+1, R1[1], R1[1]],'y')
                        im3, = plt.fill([R2[0], R2[0], R2[0]+1, R2[0]+1, R2[0]],
                                        [R2[1], R2[1]+1, R2[1]+1, R2[1], R2[1]],'y')
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
                    if o[s]==0:
                        c = 'c'
                    elif o[s]==1:
                        c = 'lime'
                    elif o[s]==2:
                        c = 'y'         
                    im1 = plt.text(self.Environment.stateSpace[states[s],1]+0.3, self.Environment.stateSpace[states[s],0]+0.1, txt, fontsize=20, backgroundcolor=c)
                    ims.append([im1,im2,im3])
        
                ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                                repeat_delay=2000)
                ani.save(name_video)
                

    class PI_LO:
# =============================================================================
#         low level policy class for tabular parameterization
# =============================================================================
        def __init__(self, pi_lo):
            self.pi_lo = pi_lo
            self.expert = TwoRewards.Expert()
                
        def Policy(self, state, option):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_lo[stateID,:,option]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
            
    class PI_B:
# =============================================================================
#         termination policy class for tabular parameterization
# =============================================================================
        def __init__(self, pi_b):
            self.pi_b = pi_b
            self.expert = TwoRewards.Expert()
                
        def Policy(self, state, option):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_b[stateID,:,option]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
            
    class PI_HI:
# =============================================================================
#         high level policy class for tabular parameterization
# =============================================================================
        def __init__(self, pi_hi):
            self.pi_hi = pi_hi
            self.expert = TwoRewards.Expert()
                
        def Policy(self, state):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_hi[stateID,:]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
        
    
    
                 
            
            
            
            
            
            
            
            
            
            