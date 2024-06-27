import random
import math
import numpy as np
from collections import namedtuple  
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from .dwn_agent_4ly import DWA

from .ReplayBuffer import ReplayBuffer  

class DWL(object):
    def __init__(self, input_shape, num_actions, num_policies, w_learning=True, batch_size=1024,
                 epsilon=.25, epsilon_decay=.995, epsilon_min=.1, wepsilon=.99, wepsilon_decay=.9995, wepsilon_min=.1,
                 memory_size=10000, learning_rate=.01, gamma=.9, hidlyr_nodes=256, init_learn_steps_num=128, device  = None):
        
 
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies  # The number of policies is the number of
        self.num_states = self.input_shape[0] 
        self.init_learn_steps_count = 0
        self.init_learn_steps_num = init_learn_steps_num
        self.device = device

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size 

        # Learning parameters for W learning net
        self.w_learning = w_learning
        self.wepsilon = wepsilon
        self.wepsilon_decay = wepsilon_decay
        self.wepsilon_min = wepsilon_min

        # Construct Agents for each policy
        self.agents = []

        
        for i in range(self.num_policies):
            self.agents.append(DWA(input_shape=self.input_shape, num_actions=self.num_actions, batch_size=self.batch_size, epsilon=self.epsilon, 
                                    epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay, memory_size=self.memory_size, 
                                    learning_rate=self.learning_rate, gamma=self.gamma, device=self.device, hidlyr_nodes=hidlyr_nodes, w_learning=self.w_learning,
                                    wepsilon=self.wepsilon, wepsilon_decay=self.wepsilon_decay, wepsilon_min=self.wepsilon_min,)) 
                

    #
    # Action nomination function
    #
    def get_action_nomination(self, x):
        nominated_actions = []
        w_values = [] 
        for agent in self.agents:
            nominated_actions.append(agent.get_action(x))
            w_values.append(agent.get_w_value(x))

        # Try different W-policies the same logic as exploration vs explotation
        if np.random.uniform() > self.wepsilon:
            policy_sel = np.argmax(w_values)
        else: 
            policy_sel = np.random.randint(0, self.num_policies)
            self.wepsilon = max(self.wepsilon * self.wepsilon_decay, self.wepsilon_min)    
        sel_action = nominated_actions[policy_sel]
        return sel_action, policy_sel, nominated_actions

    #
    # Store experiences to all agents
    #
    def store_transition(self, s, a, rewards, s_, d, policy_sel):
        for i in range(self.num_policies):
            print(f"stored q memory in agent {i}")
            self.agents[i].store_memory(s, a, rewards[i], s_, d)
            if i != policy_sel:  # Do not store experience of the policy we selected
                print(f"stored w memory in agent {i}")
                self.agents[i].store_w_memory(s, a, rewards[i], s_, d)

    #
    # Get loss values for Q and W
    #
    def get_loss_values(self):
        q_loss, w_loss, = [], []
        for i in range(self.num_policies):
            q_loss_part, w_loss_part = self.agents[i].collect_loss_info()
            q_loss.append(q_loss_part), w_loss.append(w_loss_part)
        #print(q_loss, w_loss)
        return q_loss, w_loss

    #
    # Train Q and W networks
    #
    def learn(self):
        for i in range(self.num_policies):
            print("training pol :", i)
            self.agents[i].train()
            if self.init_learn_steps_count == self.init_learn_steps_num: # we start training W-network with delay
                self.agents[i].learn_w()
        self.init_learn_steps_count += 1

    #
    # Updating Beta and exploration rate
    #
    def update_params(self):
        for i in range(self.num_policies):
            self.agents[i].update_params()

    #
    # Save trained Q-networks and W-networks to file
    #
    def save(self, path):
        for i in range(self.num_policies):
            self.agents[i].save_net(path + 'Q' + str(i) + '.pt')
            self.agents[i].save_w_net(path + 'W' + str(i) + '.pt')

    #
    # Load the pre-trained Q-networks and W-networks from file
    #
    def load(self, path):
        for i in range(self.num_policies):
            print("Loading", i)
            self.agents[i].load_net(path + 'Q' + str(i) + '.pt')
            self.agents[i].load_w_net(path + 'W' + str(i) + '.pt')
