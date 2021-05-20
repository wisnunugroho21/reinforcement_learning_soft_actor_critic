import torch
import torch.nn as nn
import numpy as np
import copy
import time
import datetime
import gym
from torch.utils.data import Dataset
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader

def set_device(use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def to_numpy(datas, use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return datas.detach().cpu().numpy()
        else:
            return datas.detach().numpy()
    else:
        return datas.detach().numpy()

def to_tensor(datas, use_gpu = True, first_unsqueeze = False, last_unsqueeze = False, detach = False):
    if isinstance(datas, tuple):
        datas = list(datas)
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                data    = data.unsqueeze(0)
            if last_unsqueeze:
                data    = data.unsqueeze(-1) if data.shape[-1] != 1 else data
            if detach:
                data    = data.detach()
            datas[i] = data
        datas = tuple(datas)

    elif isinstance(datas, list):
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                data    = data.unsqueeze(0)
            if last_unsqueeze:
                data    = data.unsqueeze(-1) if data.shape[-1] != 1 else data
            if detach:
                data    = data.detach()
            datas[i] = data
        datas = tuple(datas)

    else:
        datas   = torch.FloatTensor(datas).to(set_device(use_gpu))
        if first_unsqueeze: 
            datas   = datas.unsqueeze(0)
        if last_unsqueeze:
            datas   = datas.unsqueeze(-1) if datas.shape[-1] != 1 else datas
        if detach:
            datas   = datas.detach()
    
    return datas

def copy_parameters(source_model, target_model, tau = 0.95):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))

    return target_model

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
        ).float().to(set_device(use_gpu))

        self.actor_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Tanh()
        ).float().to(set_device(use_gpu))

        self.actor_std_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Sigmoid()
        ).float().to(set_device(use_gpu))
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      if detach:
        return (self.actor_layer(x[:, :32]).detach(), self.actor_std_layer(x[:, 32:64]).detach())
      else:
        return (self.actor_layer(x[:, :32]), self.actor_std_layer(x[:, 32:64]))
      
class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Q_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim + action_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        ).float().to(set_device(use_gpu))
        
    def forward(self, states, actions, detach = False):
      x   = torch.cat((states, actions), -1)

      if detach:
        return self.nn_layer(x).detach()
      else:
        return self.nn_layer(x)

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity
        self.position       = 0

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx]), torch.FloatTensor(self.actions[idx]), torch.FloatTensor([self.rewards[idx]]), \
            torch.FloatTensor([self.dones[idx]]), torch.FloatTensor(self.next_states[idx])

    def save_eps(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, rewards, dones, next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states 

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

class BasicContinous():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def sample(self, datas):
        mean, std = datas

        distribution    = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample().float().to(set_device(self.use_gpu))
        return (mean + std.squeeze() * rand).squeeze(0)
        
    def entropy(self, datas):
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(set_device(self.use_gpu))
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        # return distribution.log_prob(value_data).float().to(set_device(self.use_gpu))

        bounded_value_data = torch.tanh(value_data)
        return distribution.log_prob(value_data) - torch.log(1 - bounded_value_data.pow(2) + 1e-6)

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2).float().to(set_device(self.use_gpu))

    def deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)

class PolicyLoss():
    def __init__(self, distribution):
        self.distribution       = distribution

    def compute_loss(self, action_datas, actions, predicted_q_value1, predicted_q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        policy_loss             = (torch.min(predicted_q_value1, predicted_q_value2) - 0.2 * log_prob).mean()
        return policy_loss * -1

class QLoss():
    def __init__(self, distribution, gamma = 0.99):
        self.gamma          = gamma
        self.distribution   = distribution

    def compute_loss(self, predicted_q_value1, predicted_q_value2, target_q_value1, target_q_value2, next_action_datas, next_actions, reward, done):
        next_log_prob           = self.distribution.logprob(next_action_datas, next_actions)
        next_value              = (torch.min(target_q_value1, target_q_value2) - 0.2 * next_log_prob).detach()

        target_q_value          = (reward + (1 - done) * self.gamma * next_value).detach()

        q_value_loss1           = ((target_q_value - predicted_q_value1).pow(2) * 0.5).mean()
        q_value_loss2           = ((target_q_value - predicted_q_value2).pow(2) * 0.5).mean()
        
        return q_value_loss1 + q_value_loss2

class AgentSAC():
    def __init__(self, soft_q1, soft_q2, policy, state_dim, action_dim, distribution, q_loss, policy_loss, memory, 
        soft_q_optimizer, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', use_gpu = True):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy

        self.soft_q1            = soft_q1
        self.target_soft_q1     = copy.deepcopy(self.soft_q1)

        self.soft_q2            = soft_q2
        self.target_soft_q2     = copy.deepcopy(self.soft_q2)             

        self.distribution       = distribution
        self.memory             = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer

        self.soft_q_scaler      = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()        

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            next_action_datas   = self.policy(next_states, True)
            next_actions        = self.distribution.sample(next_action_datas).detach()

            target_q_value1     = self.target_soft_q1(next_states, torch.tanh(next_actions), True)
            target_q_value2     = self.target_soft_q2(next_states, torch.tanh(next_actions), True)

            predicted_q_value1  = self.soft_q1(states, actions)
            predicted_q_value2  = self.soft_q2(states, actions)

            loss  = self.qLoss.compute_loss(predicted_q_value1, predicted_q_value2, target_q_value1, target_q_value2, next_action_datas, next_actions, rewards, dones)

        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas        = self.policy(states)
            actions             = self.distribution.sample(action_datas)

            predicted_q_value1  = self.soft_q1(states, torch.tanh(actions))
            predicted_q_value2  = self.soft_q2(states, torch.tanh(actions))

            loss = self.policyLoss.compute_loss(action_datas, actions, predicted_q_value1, predicted_q_value2)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_sac(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                dataloader  = DataLoader(self.memory, self.batch_size, shuffle = True, num_workers = 2)
                states, actions, rewards, dones, next_states = next(iter(dataloader))

                self._training_q(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), next_states.float().to(self.device))
                self._training_policy(states.float().to(self.device))

            self.target_soft_q1 = copy_parameters(self.soft_q1, self.target_soft_q1, self.soft_tau)
            self.target_soft_q2 = copy_parameters(self.soft_q2, self.target_soft_q2, self.soft_tau)

    def update(self):
        self._update_sac()

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        state               = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        action = torch.tanh(action)
        return to_numpy(action, self.use_gpu)    

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
            'policy_scaler_state_dict': self.policy_scaler.state_dict(),
            'soft_q_scaler_state_dict': self.soft_q_scaler.state_dict(),
        }, self.folder + '/sac.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])
        self.soft_q1.load_state_dict(model_checkpoint['soft_q1_state_dict'])
        self.soft_q2.load_state_dict(model_checkpoint['soft_q2_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict'])
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])

class SingleStepRunner():
    def __init__(self, agent, env, memory, training_mode, render, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.agent              = agent
        self.env                = env
        self.memories           = memory

        self.render             = render
        self.training_mode      = training_mode
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.is_discrete        = is_discrete

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self):
        self.memories.clear_memory() 
        action  = self.agent.act(self.states)

        if self.is_discrete:
            action = int(action)

        if self.max_action is not None and not self.is_discrete:
            action_gym  =  np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, _ = self.env.step(action_gym)
        else:
            next_state, reward, done, _ = self.env.step(action)
        
        if self.training_mode:
            self.memories.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
            
        self.states         = next_state
        self.eps_time       += 1 
        self.total_reward   += reward
                
        if self.render:
            self.env.render()

        if done:                
            self.i_episode  += 1
            print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

            if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                self.writer.add_scalar('Times', self.eps_time, self.i_episode)

            self.states         = self.env.reset()
            self.total_reward   = 0
            self.eps_time       = 0        

        return self.memories

class Executor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode 
        self.load_weights       = load_weights       

    def execute(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                memories  = self.runner.run()
                self.agent.save_memory(memories)

                if self.is_training_mode:
                    self.agent.update()

                    if self.save_weights:
                        if i_iteration % self.n_saved == 0:
                            self.agent.save_weights()
                            print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

class GymWrapper():
    def __init__(self, env):
        self.env = env        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.env.observation_space.shape) > 1:                
                for i in range(len(self.env.observation_space.shape)):
                    state_dim *= self.env.observation_space.shape[i]            
            else:
                state_dim = self.env.observation_space.shape[0]

            return state_dim
        else:
            return self.env.observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

import random
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_iteration             = 1000000
n_plot_batch            = 1
soft_tau                = 0.95
n_saved                 = 1
epochs                  = 1
batch_size              = 32
learning_rate           = 3e-4

folder                  = 'weights/carla'
env                     = gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

Policy_Model        = Policy_Model
Q_Model             = Q_Model
Policy_Dist         = BasicContinous
Runner              = SingleStepRunner
Executor            = Executor
Policy_loss         = PolicyLoss
Q_loss              = QLoss
Wrapper             = GymWrapper
Policy_Memory       = PolicyMemory
Agent               = AgentSAC

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment = Wrapper(env)

if state_dim is None:
    state_dim = environment.get_obs_dim()
print('state_dim: ', state_dim)

if environment.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = environment.get_action_dim()
print('action_dim: ', action_dim)

policy_dist         = Policy_Dist(use_gpu)
sac_memory          = Policy_Memory()
runner_memory       = Policy_Memory()
q_loss              = Q_loss(policy_dist)
policy_loss         = Policy_loss(policy_dist)

policy              = Policy_Model(state_dim, action_dim, use_gpu).float().to(set_device(use_gpu))
soft_q1             = Q_Model(state_dim, action_dim).float().to(set_device(use_gpu))
soft_q2             = Q_Model(state_dim, action_dim).float().to(set_device(use_gpu))
policy_optimizer    = Adam(list(policy.parameters()), lr = learning_rate)
soft_q_optimizer    = Adam(list(soft_q1.parameters()) + list(soft_q2.parameters()), lr = learning_rate)

agent = Agent(soft_q1, soft_q2, policy, state_dim, action_dim, policy_dist, q_loss, policy_loss, sac_memory, 
        soft_q_optimizer, policy_optimizer, is_training_mode, batch_size, epochs, 
        soft_tau, folder, use_gpu)
                    
runner      = Runner(agent, environment, runner_memory, is_training_mode, render, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()