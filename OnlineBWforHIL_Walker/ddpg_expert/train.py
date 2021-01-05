import gym
import torch
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v3')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)


def ddpg(episodes, step, pretrained, noise):

    if pretrained:
        agent.actor_local.load_state_dict(torch.load('weights-simple/checkpoint_actor_t.pth', map_location="cpu"))
        agent.critic_local.load_state_dict(torch.load('weights-simple/checkpoint_critic_t.pth', map_location="cpu"))
        agent.actor_target.load_state_dict(torch.load('weights-simple/checkpoint_actor_t.pth', map_location="cpu"))
        agent.critic_target.load_state_dict(torch.load('weights-simple/checkpoint_critic_t.pth', map_location="cpu"))

    reward_list = []
    Reward_array = np.empty((0))
    obs = env.reset()
    size_input = len(obs)
    TrainingSet = np.empty((0,size_input))
            
    action = np.array([0.0, 0.0, 0.0, 0.0])
    size_action = len(action)
    Labels = np.empty((0,size_action))    

    for i in range(episodes):

        obs = np.round(env.reset(),2)
        score = 0
        Reward = 0
        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
        # Discretize state into buckets
        done = False
                
        # policy action 
        Labels = np.append(Labels, action.reshape(1,size_action),0)

        for t in range(step):

            env.render()

            action = np.round(agent.act(obs, noise),0)
            Labels = np.append(Labels, action.reshape(1,size_action),0)
            obs, reward, done, info = env.step(action[0])
            obs = np.round(obs,2)
            TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    
            Reward = Reward + reward
            obs = obs.squeeze()
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(score, i, episodes))
                break

        reward_list.append(score)

        if score >= 270:
            print('Task Solved')
            torch.save(agent.actor_local.state_dict(), 'weights-simple/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights-simple/checkpoint_critic.pth')
            torch.save(agent.actor_target.state_dict(), 'weights-simple/checkpoint_actor_t.pth')
            torch.save(agent.critic_target.state_dict(), 'weights-simple/checkpoint_critic_t.pth')
            break

    torch.save(agent.actor_local.state_dict(), 'weights-simple/checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'weights-simple/checkpoint_critic.pth')
    torch.save(agent.actor_target.state_dict(), 'weights-simple/checkpoint_actor_t.pth')
    torch.save(agent.critic_target.state_dict(), 'weights-simple/checkpoint_critic_t.pth')

    print('Training saved')
    env.close()
    
    return TrainingSet, Labels, reward_list


TrainingSet, Labels, scores = ddpg(episodes=20, step=2000, pretrained=1, noise=0)

fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

with open('DataFromExpert/TrainingSet.npy', 'wb') as f:
    np.save(f, TrainingSet)
    
with open('DataFromExpert/Labels.npy', 'wb') as f:
    np.save(f, Labels)
    
with open('DataFromExpert/Reward.npy', 'wb') as f:
    np.save(f, scores)
    
    