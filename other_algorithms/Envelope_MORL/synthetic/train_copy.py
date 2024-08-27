from __future__ import absolute_import, division, print_function
import argparse
import time
import numpy as np
import pandas as pd
import torch
import os
import pickle 
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv
from pyews.server_interface import ewsRESTInterface as eRI 
from pyews.global_vars import settings  

parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-envelope', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=1000000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.99, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=True, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=600, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='./other_algorithms/Envelope_MORL/synthetic/crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='./other_algorithms/Envelope_MORL/synthetic/crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

definitions = {
    "URL" : "http://localhost:2011/",    
    "main_component" : "../repository/TCPNetwork.o",
    "proxy_JSON" : {"exp":"|../pal/monitoring/proxies/HTTPProxy.o|*(*:http.handler.GET.HTTPGET[0]:*)|"}
} 

settings["IP"] = definitions["URL"]

COLLECTION_WINDOW = 3
#COLLECTION_WINDOW = 0.5
DESIRED_SAMPLES = 10
CHOSEN_METRIC = "response_time"
CONFIG = 0
REWARD = 1
N_K = 2
COST = 3

# Simulation Parameters
REPEATS = 100
EPISODES = 600

configurations = []
cost_dict = {}
labels = []  

configurations = eRI.get_all_configs() 
confs_static = configurations

        #cost_dict = {config: np.random.rand() for config in configurations} 
        #labels = [ [config , cost_dict[config]] for config in configurations]  

with open('./results/labels_cost_static.pkl', 'rb') as file:
        try:
            labels = pickle.load(file)
        except AttributeError as e:
            print(f"Attribute error encountered: {e}") 
            raise 
for item in labels:
        ind = labels.index(item)
        labels[ind][0]= configurations[labels[ind][0]]
        cost_dict[labels[ind][0]] = item[1] 
        
np.random.shuffle(configurations)
knowledge = [[config, 0, 1, 0] for config in configurations] #[config, cumulative reward, times_chosen, peak response time]


    # Get environment information
state = []
count = 0
for sublist in knowledge:
        state.append(count)
        count+=1
state = np.asarray(state) 
state_shape = state.shape  
action_num = len(knowledge)
num_objectives = 2

def truncate_normalize(cost, preferHigh):
    upper_bound = 300
    lower_bound = 0
 #truncate
    if(cost > upper_bound):
        cost = upper_bound
    elif(cost < lower_bound):
        cost = lower_bound


    cost_range = upper_bound - lower_bound
    result = float((cost - lower_bound)/cost_range) #normalize
    
    if(not preferHigh):
        result = 1.0 - result

    return result


def train(env, agent, args):
    for repeat in range(REPEATS):
        monitor = Monitor(train=True, spec="-{}".format(args.method))
        monitor.init_log(args.log, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
        env.reset()
        
        configurations = []
        cost_dict = {}
        labels = []  

        configurations = eRI.get_all_configs() 
        confs_static = configurations

            #cost_dict = {config: np.random.rand() for config in configurations} 
            #labels = [ [config , cost_dict[config]] for config in configurations]  

        with open('./results/labels_cost_static.pkl', 'rb') as file:
            try:
                labels = pickle.load(file)
            except AttributeError as e:
                print(f"Attribute error encountered: {e}") 
                raise 
        for item in labels:
            ind = labels.index(item)
            labels[ind][0]= configurations[labels[ind][0]]
            cost_dict[labels[ind][0]] = item[1] 
            
        np.random.shuffle(configurations)
        knowledge = [[config, 0, 1, 0] for config in configurations] #[config, cumulative reward, times_chosen, peak response time]


        # Get environment information
        state = []
        count = 0
        for sublist in knowledge:
            state.append(count)
            count+=1
        state = np.asarray(state) 
        state_shape = state.shape  
        action_num = len(knowledge)
        num_objectives = 2
        print("Information about environment, state shape:", state_shape, " action space:", action_num,
                "number of objective:", num_objectives)
        


        num_of_steps = [] 

        avg_response_time = []
        costs = []
        chosen_confs = []
        chosen_confs_indexes = []
        requestCounter = []


        for num_eps in range(args.episode_num):
            terminal = False
            env.reset()
            loss = 0
            cnt = 0
            tot_reward = 0

            done = False  
            num_of_requests = 0
            qSum = 0
            qActions = 1
            lossSum = 0 
            num_steps = 0  
            num_steps += 1 

            probe = None
            if args.env_name == "dst":
                probe = torch.tensor([0.8, 0.2])
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                probe = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])

            probe = torch.tensor([0.8, 0.2]).to(device)

            #Action Selection
            nom_action = agent.act(state)

            #accion Execution and evironment evolution

            new_configuration_i = knowledge[nom_action]
            print(f"\n\n\n\n index : {nom_action} conf: {new_configuration_i[0]}")
            eRI.change_configuration(configurations[nom_action])
            chosen_confs_indexes.append(nom_action)
            chosen_confs.append(configurations[nom_action])
            print("new config created")
            cost_of_episode = -cost_dict[configurations[nom_action]]
            start_time = time.time()
            sample_list = []
            reqCount = 0
            while (time.time() - start_time) < COLLECTION_WINDOW:
                        perception = eRI.get_perception() 
                        reading = None

                        if(CHOSEN_METRIC in perception.metric_dict):
                            reading = perception.metric_dict[CHOSEN_METRIC] 
                        if(reading):
                            val = truncate_normalize(reading.value,reading.is_preference_high)
                            sample_list.extend([val])
                            reqCount+=1
                        #else:
                        #    print("There is no traffic being experienced by the EWS and not enough samples have been collected yet")  
            knowledge[nom_action][REWARD] += sum(sample_list) 
            knowledge[nom_action][N_K] += reqCount
            knowledge[nom_action][COST] = cost_of_episode

            nextState = state
            avg_response_time_of_episode = -(sum(sample_list) / reqCount)
            if avg_response_time_of_episode < -0.5 :
                    avg_response_time_of_episode = -2
                    cost_of_episode = -2
            if cost_of_episode < -0.5 : 
                    avg_response_time_of_episode = -2
                    cost_of_episode = -2 

            reward = [avg_response_time_of_episode, cost_of_episode] 
            reward = np.array(reward)
            requestCounter.append(reqCount)
            reqCount=0 
            next_state = nextState
            terminal = True
            print(reward)
                #logging
            if args.log:
                    monitor.add_log(state, nom_action, reward, terminal, agent.w_kept)

                #add memory and training
            agent.memorize(state, nom_action, next_state, reward, terminal)
            loss += agent.learn()
            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)

            _, q = agent.predict(probe)

            if args.env_name == "dst":
                act_1 = q[0, 3]
                act_2 = q[0, 1]
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                act_1 = q[0, 1]
                act_2 = q[0, 0]

            act_1 = q[0, 1]
            act_2 = q[0, 0]

            if args.method == "crl-naive":
                act_1 = act_1.data.cpu()
                act_2 = act_2.data.cpu()
            elif args.method == "crl-envelope":
                act_1 = probe.dot(act_1.data)
                act_2 = probe.dot(act_2.data)
            elif args.method == "crl-energy":
                act_1 = probe.dot(act_1.data)
                act_2 = probe.dot(act_2.data)

            # Save the performance to lists
            num_of_steps.append(num_steps) 

            avg_response_time.append(-avg_response_time_of_episode)
            costs.append(-cost_of_episode)
            
            #avg loss : 
            if loss == 0 :
                 avg_loss = 0
            else :
                 avg_loss = loss / num_eps+1

            print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f; loss: %0.4f" % (
                num_eps,
                tot_reward,
                act_1,
                act_2,
                # q__max,
                avg_loss))
            monitor.update(num_eps,
                        tot_reward,
                        act_1,
                        act_2,
                        #    q__max,
                        avg_loss)
        # if num_eps+1 % 100 == 0:
        # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))


        df_results = pd.DataFrame()
        df_results['episodes'] = range(1, EPISODES + 1) 
        df_results["chosen_conf_index"] = chosen_confs_indexes
        df_results["chosen_conf"] = chosen_confs
        df_results["chosen_conf_index_in_labels"] = df_results["chosen_conf"].apply(lambda x: labels.index([x,cost_dict[x]]))
        df_results["number_of_requests"] = requestCounter 

        df_results['avg_response_time'] = avg_response_time
        df_results['cost'] = costs


        with open(f"./results/envelope/labels_env_{repeat}.pkl", 'wb+') as f:
                labels = [[confs_static.index(label[0]), label[1]] for label in labels]
                pickle.dump(labels, f)
        f.close()

        df_results.to_csv(f"./results/envelope/ews_env_{repeat}.csv")

        agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # get state / action / reward sizes
    state_size = len(state)
    action_size = action_num
    reward_size = num_objectives

    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        model = get_new_model(args.model, state_size, action_size, reward_size)
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
