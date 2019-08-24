import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedEnv(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
              ).float().to(device)
        
    def forward(self, state):
        return self.value_layer(state)  

    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param)

    def lets_init_weights(self):      
        self.value_layer.apply(self.init_weights)
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(SoftQNetwork, self).__init__()

        self.softQ_layer = nn.Sequential(
                nn.Linear(num_inputs  + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
              ).float().to(device)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.softQ_layer(x) 

    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param)

    def lets_init_weights(self):      
        self.softQ_layer.apply(self.init_weights)
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.policy_layer = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
              ).float().to(device)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)        
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        x = self.policy_layer(state)
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param)

    def lets_init_weights(self):      
        self.policy_layer.apply(self.init_weights)

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim = 256, replay_buffer_size = 1000000, batch_size = 128, epsilon = 1e-6, learning_rate = 3e-4):
        self.gamma    = 0.99
        self.soft_tau = 0.001
        self.epochs = 1
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.epsilon = epsilon        

        self.value_net         = ValueNetwork(state_dim, hidden_dim).to(device)
        self.target_value_net  = ValueNetwork(state_dim, hidden_dim).to(device)
        self.soft_q_net1       = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2       = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net        = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.value_optimizer   = optim.Adam(self.value_net.parameters(), lr = learning_rate)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr = learning_rate)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr = learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

    def prepro(self, mean, log_std, epsilon): 
        std = log_std.exp()

        normal    = Normal(0, 1)
        z         = normal.sample()
        action    = torch.tanh(mean + std * z.to(device))
        log_prob  = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()
        return action[0]

    def get_Q_loss(self, reward, done, target_value, predicted_q_value):
        target_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = (predicted_q_value - target_q_value.detach()).pow(2).mean()
        return q_value_loss

    def get_V_loss(self, predicted_new_q_value1, predicted_new_q_value2, log_prob, predicted_value):
        predicted_new_q_value = torch.min(predicted_new_q_value1, predicted_new_q_value2)
        target_value_func = predicted_new_q_value - log_prob
        value_loss = (predicted_value - target_value_func.detach()).pow(2).mean()
        return value_loss

    def get_policy_loss(self, predicted_new_q_value1, predicted_q_value2, log_prob):
        predicted_new_q_value = torch.min(predicted_new_q_value1, predicted_q_value2)
        policy_loss = (log_prob - predicted_new_q_value).mean()
        return policy_loss

    def update(self):
        for i in range(self.epochs):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            state      = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action     = torch.FloatTensor(action).to(device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

            predicted_q_value1  = self.soft_q_net1(state, action)
            predicted_q_value2  = self.soft_q_net2(state, action)
            predicted_value     = self.value_net(state)
            mean, log_std       = self.policy_net(state)

            target_value            = self.target_value_net(next_state)
            new_action, log_prob, _, mean, log_std = self.prepro(mean, log_std, self.epsilon)
            predicted_new_q_value1  = self.soft_q_net1(state, new_action)
            predicted_new_q_value2  = self.soft_q_net2(state, new_action)        

            q_value_loss1 = self.get_Q_loss(reward, done, target_value, predicted_q_value1)
            q_value_loss2 = self.get_Q_loss(reward, done, target_value, predicted_q_value2)
            value_loss    = self.get_V_loss(predicted_new_q_value1, predicted_new_q_value2, log_prob, predicted_value)
            policy_loss   = self.get_policy_loss(predicted_new_q_value1, predicted_new_q_value2, log_prob)

            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            self.soft_q_optimizer1.step()

            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            self.soft_q_optimizer2.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def lets_init_weights(self):
        self.value_net.lets_init_weights()
        self.soft_q_net1.lets_init_weights()
        self.soft_q_net2.lets_init_weights()
        self.policy_net.lets_init_weights()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

    def load_weights(self):
        self.value_net.load_state_dict(torch.load('/test/My Drive/RL_Bipedal_SAC/value_net.pth'))        
        self.soft_q_net1.load_state_dict(torch.load('/test/My Drive/RL_Bipedal_SAC/soft_q_net1.pth'))
        self.soft_q_net2.load_state_dict(torch.load('/test/My Drive/RL_Bipedal_SAC/soft_q_net2.pth'))        
        self.policy_net.load_state_dict(torch.load('/test/My Drive/RL_Bipedal_SAC/policy_net.pth'))
        self.target_value_net.load_state_dict(torch.load('/test/My Drive/RL_Bipedal_SAC/target_value_net.pth'))

    def save_weights(self):
        torch.save(self.value_net.state_dict(), '/test/My Drive/RL_Bipedal_SAC/value_net.pth')
        torch.save(self.soft_q_net1.state_dict(), '/test/My Drive/RL_Bipedal_SAC/soft_q_net1.pth')
        torch.save(self.soft_q_net2.state_dict(), '/test/My Drive/RL_Bipedal_SAC/soft_q_net2.pth')
        torch.save(self.policy_net.state_dict(), '/test/My Drive/RL_Bipedal_SAC/policy_net.pth')
        torch.save(self.target_value_net.state_dict(), '/test/My Drive/RL_Bipedal_SAC/target_value_net.pth')

def plot(datas):
    print('----------')
    
    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()
    
    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def run_episode(env, agent, state_dim, render, training_mode):
    ############################################
    state = env.reset()
    done = False
    total_reward = 0
    t = 0
    ############################################
    
    while not done:      
        action = agent.act(state).detach()
        next_state, reward, done, _ = env.step(action.numpy())
          
        total_reward += reward 
        reward *= 10
        
        if training_mode:
            agent.replay_buffer.push(state, action, reward, next_state, done) 
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
        state = next_state
        t += 1              
                
        if render:
            env.render()
        
    return total_reward, t

def run_init_explore(env, agent, render, max_init_explore):
    ############################################
    state = env.reset()
    done = False
    ############################################
    
    for i in range(max_init_explore):    
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
          
        agent.replay_buffer.push(state, action, reward, next_state, done) 
        if len(agent.replay_buffer) > agent.batch_size:
            agent.update()
            
        state = next_state 
                
        if render:
            env.render()

        if done:
            state = env.reset()
        
    return agent
    
def main():
    ############## Hyperparameters ##############
    using_google_drive = True # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = False # If you want to load the agent, set this to True
    save_weights = True # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_episode = 1000 # How many episode you want to run
    max_init_explore = 200
    #############################################         
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
        
    agent = Agent(state_dim, action_dim, hidden_dim = 512, batch_size = 256)  
    ############################################# 
    
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')
    
    if load_weights:
        agent.load_weights()
        print('Weight Loaded')
    else :
        agent.lets_init_weights()
        print('Init Weight')
    
    if torch.cuda.is_available() :
        print('Using GPU')
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []

    #############################################

    if training_mode:
        agent = run_init_explore(env, agent, render, max_init_explore)

    #############################################    
    
    for i_episode in range(1, n_episode):
        total_reward, time = run_episode(env, agent, state_dim, render, training_mode)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(total_reward)
        batch_times.append(time)       
        
        if training_mode:
            # update after n episodes
            if save_weights:
                agent.save_weights()
                print('Weights saved')
                    
        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold :              
                    for reward in batch_times:
                        rewards.append(reward)

                    for time in batch_rewards:
                        times.append(time)                    

                    print('You solved task after {} episode'.format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)
            
        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)
            
            for reward in batch_rewards:
                rewards.append(reward)
                
            for time in batch_times:
                times.append(time)
                
            batch_rewards = []
            batch_times = []

            print('========== Cummulative ==========')
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)
            
    print('========== Final ==========')
    # Plot the reward, times for every episode
    plot(rewards)
    plot(times) 
            
if __name__ == '__main__':
    main()