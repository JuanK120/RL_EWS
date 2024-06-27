import copyreg
import os
import pickle 
from pyews.server_interface import ewsRESTInterface as eRI 
from pyews.global_vars import settings  
import time
import pandas as pd
import numpy as np
import math
import collections
import torch 
from agents.dqn_agent_4ly import DQN 
import torch 

COLLECTION_WINDOW = 3
DESIRED_SAMPLES = 10
CHOSEN_METRIC = "response_time"
CONFIG = 0
REWARD = 1
N_K = 2
COST = 3

# The Q-learning agent parameters
BATCH_SIZE = 64
LR = 0.01                  # learning rate
EPSILON = .99           # starting epsilon for greedy policy
EPSILON_MIN = .0001           # The minimal epsilon we want
EPSILON_DECAY = .99      # The minimal epsilon we want 
GAMMA = .99                # reward discount
MEMORY_SIZE = 1000000        # size of the replay buffer

# Simulation Parameters
REPEATS = 8
EPISODES = 600 

definitions = {
    "URL" : "http://localhost:2011/",    
    "main_component" : "../repository/TCPNetwork.o",
    "proxy_JSON" : {"exp":"|../pal/monitoring/proxies/HTTPProxy.o|*(*:http.handler.GET.HTTPGET[0]:*)|"}
} 

settings["IP"] = definitions["URL"]

#eRI.initialize_server(definitions["main_component"],definitions["proxy_JSON"])

# Path to trained Net, without extensions as used in the per net...
PATH = 'agents/savedNets/ews_wl/ews_ly_dqn'  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


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

if __name__ == "__main__": 
    
    for repeat in range(REPEATS):

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
        # Init the W-learning agent
        agent = DQN(state_shape, action_num, num_objectives, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                    epsilon_decay=EPSILON_DECAY, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, 
                    learning_rate=LR, gamma=GAMMA, hidlyr_nodes=256, device=device)
        # Init list for information we need to collect during simulation
        
        num_of_steps = []
        if os.path.exists(PATH):
            #agent.load(PATH)
            print("here")
        else:
            print("saved net does not exist!")
        
        coll_reward1 = []  

        loss_q1_episode = []  

        avg_response_time = []
        costs = []
        chosen_confs = []
        chosen_confs_indexes = []
        requestCounter = []

        for episode in range(EPISODES):
            
            done = False  

            num_of_requests = 0

            qSum = 0

            qActions = 1

            lossSum = 0 

            num_steps = 0 

            nom_action = agent.get_action(state)
            num_steps += 1 

            new_configuration_i = knowledge[nom_action]

            print(f"\n\n\n\n index : {nom_action} conf: {new_configuration_i[0]}")
            eRI.change_configuration(configurations[nom_action])
            chosen_confs_indexes.append(nom_action)
            chosen_confs.append(configurations[nom_action])
            print("new config created")
            cost_of_episode = cost_dict[configurations[nom_action]]
            start_time = time.time()
            sample_list = []
            reqCount = 0

            while (time.time() - start_time) < COLLECTION_WINDOW:
                perception = eRI.get_perception() 
                reading = None

                if(CHOSEN_METRIC in perception.metric_dict):
                    reading = perception.metric_dict[CHOSEN_METRIC]
                
                if(reading):
                    sample_list.extend([truncate_normalize(reading.value,reading.is_preference_high)])
                    reqCount+=1
                #else:
                #    print("There is no traffic being experienced by the EWS and not enough samples have been collected yet") 

            done = True          

            knowledge[nom_action][REWARD] += sum(sample_list) 
            knowledge[nom_action][N_K] += reqCount
            knowledge[nom_action][COST] = cost_of_episode

            nextState = state

            avg_response_time_of_episode = sum(sample_list) / reqCount

            if avg_response_time_of_episode > 0.5 :
                avg_response_time_of_episode = 2
                cost_of_episode = 2

            if cost_of_episode > 0.5 : 
                avg_response_time_of_episode = 2
                cost_of_episode = 2 

            reward = -avg_response_time_of_episode + -cost_of_episode 

            requestCounter.append(reqCount)
            reqCount=0 

            nextState = nextState 

            agent.store_memory(state, nom_action,reward, nextState, done) 

            agent.train() 

            state = nextState
            done=True
            
            q_loss = agent.collect_loss_info()
            print("Episode", episode,"avg. time", avg_response_time_of_episode,"cost", cost_of_episode, "end_reward", reward, "Num steps:", num_steps,
                "Epsilon:", agent.epsilon, "Q loss:", q_loss ) 
            
            # Save the performance to lists
            num_of_steps.append(num_steps)

            coll_reward1.append(reward)     

            loss_q1_episode.append(q_loss)  

            avg_response_time.append(avg_response_time_of_episode)
            costs.append(cost_of_episode)



            agent.update_params()

        # Save the results
        df_results = pd.DataFrame()
        df_results['episodes'] = range(1, EPISODES + 1) 
        df_results["chosen_conf_index"] = chosen_confs_indexes
        df_results["chosen_conf"] = chosen_confs
        df_results["chosen_conf_index_in_labels"] = df_results["chosen_conf"].apply(lambda x: labels.index([x,cost_dict[x]]))
        df_results["number_of_requests"] = requestCounter
        df_results['rewardsum'] = coll_reward1   

        df_results['loss_q1'] = loss_q1_episode   

        df_results['avg_response_time'] = avg_response_time
        df_results['cost'] = costs


        with open(f"./results/dqn/labels_dqn_{repeat}.pkl", 'wb+') as f:
            labels = [[confs_static.index(label[0]), label[1]] for label in labels]
            pickle.dump(labels, f)
        f.close()

        df_results.to_csv(f"./results/dqn/ews_dqn_{repeat}.csv")
        # Save the trained ANN
        agent.save_net(PATH+f"_{repeat}.pt") 





