import random
import math
import numpy as np
from collections import namedtuple  
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from .ReplayBuffer import ReplayBuffer


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QN (nn.Module):
    """
        Basic Model of a Q network
    """
    
    def __init__(self, state_number, hidlyr_nodes, action_number):
        super(QN,self).__init__()
        self.fc1 = nn.Linear(state_number,hidlyr_nodes) #first conected layer
        self.fc2 = nn.Linear(hidlyr_nodes,hidlyr_nodes*2) #second conected layer
        self.fc3 = nn.Linear(hidlyr_nodes*2,hidlyr_nodes*4) #third conected layer
        self.fc4 = nn.Linear(hidlyr_nodes*4,hidlyr_nodes*8) #fourth conected layer
        self.out = nn.Linear(hidlyr_nodes*8,action_number) #output layer

    def forward(self,state):
        x = F.relu(self.fc1(state)) #relu activation of fc1
        x = F.relu(self.fc2(x)) #relu activation of fc2
        x = F.relu(self.fc3(x)) #relu activation of fc3
        x = F.relu(self.fc4(x)) #relu activation of fc4
        x = self.out(x)  #calculate output
        return x
   
class DQN(object):
    def __init__(self, input_shape, num_actions, net_sync_rate=1024, batch_size=1024, epsilon=.25,
                 epsilon_decay=.9999, epsilon_min=.1, tau=.001, memory_size=10000, learning_rate=.01,
                 gamma=.9, per_epsilon=.001, beta_start=.4, beta_inc=1.002, seed=404, device = None, hidlyr_nodes=128):
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_states = self.input_shape[0]

        self.random_seed = seed

        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        self.replayMemory = ReplayBuffer(self.num_actions, self.memory_size, self.batch_size, self.random_seed)

        self.per_epsilon = per_epsilon 

        self.q_episode_loss = []

        self.policy_net = QN(self.num_states,hidlyr_nodes,self.num_actions)
        self.target_net = QN(self.num_states,hidlyr_nodes,self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) 

        if device is not None : 
            self.device = device
            self.policy_net.to(device)
            self.target_net.to(device)

    def get_action(self,x):
        if type(x).__name__ == 'ndarray':
            state = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        else:
            state = x
        a = self.policy_net.eval() 
        with torch.no_grad():
            action_values = self.policy_net(state)
 
        self.policy_net.train() 

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else: 
            return random.choice(np.arange(self.num_actions))
        
    def store_memory (self, state, action, reward, next_state, done):
        self.replayMemory.add(state,action,reward,next_state,done)

    def train(self):
        if len(self.replayMemory) > self.batch_size:  
            states, actions, rewards, next_states, probabilities, experiences_idx, dones = self.replayMemory.sample()
            
            current_qs = self.policy_net(states).gather(1, actions)
            next_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.target_net(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs * (1 - dones) 

            is_weights = np.power(probabilities * len(self.replayMemory), -self.beta)
            is_weights = torch.from_numpy(is_weights / is_weights.max()).float().to(self.device)
            loss = (target_qs - current_qs).pow(2) * is_weights
            loss = loss.mean()
            
            self.q_episode_loss.append(loss.detach().cpu().numpy())
            
            td_errors = (target_qs - current_qs).detach().cpu().numpy()
            self.replayMemory.update_priorities(experiences_idx, td_errors, self.per_epsilon) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.soft_update(self.policy_net, self.target_net, self.tau)
            self.update_params()
        else:
            print('memory not big enough yet \n\n')
    
    def soft_update(self, originNet, targetNet, tau):
        for target_param, local_param in zip(targetNet.parameters(), originNet.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_params(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon) 
        self.beta = min(1.0, self.beta_inc * self.beta)


    def save_net(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print('Net saved')

    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load(path)), self.target_net.load_state_dict(torch.load(path))
        self.policy_net.eval(), self.target_net.eval()

    def collect_loss_info(self): 
        avg_q_loss = np.average(self.q_episode_loss) 
        self.q_episode_loss = [] 
        return avg_q_loss 
        

