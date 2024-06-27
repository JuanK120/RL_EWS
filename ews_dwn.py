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
from agents.DWL_4ly import DWL 
import torch 

#COLLECTION_WINDOW = 3
COLLECTION_WINDOW = .5
DESIRED_SAMPLES = 10
CHOSEN_METRIC = "response_time"
CONFIG = 0
REWARD = 1
N_K = 2
COST = 3 
TRAINING_WINDOW=128


# The Q-learning agent parameters
BATCH_SIZE = 64
LR = 0.01                  # learning rate
EPSILON = .99           # starting epsilon for greedy policy
EPSILON_MIN = .001          # The minimal epsilon we want
EPSILON_DECAY = .99       
W_EPSILON = .99             # starting epsilon for greedy policy
W_EPSILON_MIN = .001         # The minimal epsilon we want
W_EPSILON_DECAY = .99      
GAMMA = .9                # reward discount
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
PATH = 'agents/savedNets/ews_wl/ews_4ly_'  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print('main device', device)

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

        with open('./results/labels_cost_static.pkl', 'rb') as file:
            try:
                labels = pickle.load(file) 
            except AttributeError as e:
                print(f"Attribute error encountered: {e}") 
                raise 
            file.close() 


        cost_dict = {} 
        for indconfigs in range(len(configurations)):
            labels[indconfigs][0] = configurations[indconfigs]
            cost_dict[configurations[indconfigs]] = labels[indconfigs][1] 
            
        #print(cost_dict)

        np.random.shuffle(configurations)

        #indofconf0 = labels.index([configurations[0], cost_dict[configurations[0]]])
        #cost_dict[configurations[0]]=100
        #labels[indofconf0][1]=100
        #print(labels[indofconf0])

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
        agent = DWL(state_shape, action_num, num_objectives, epsilon=EPSILON, epsilon_min=EPSILON_MIN, device=device,
                    epsilon_decay=EPSILON_DECAY, wepsilon=W_EPSILON, wepsilon_decay=W_EPSILON_DECAY, batch_size=BATCH_SIZE ,
                    wepsilon_min=W_EPSILON_MIN, memory_size=MEMORY_SIZE, learning_rate=LR, gamma=GAMMA, hidlyr_nodes=256, 
                    init_learn_steps_num = TRAINING_WINDOW)
        # Init list for information we need to collect during simulation
        
        num_of_steps = []

        if os.path.exists(PATH+"Q0.pt"):
            #agent.load(PATH)
            print("here")
        else:
            print("saved net does not exist!")
        
        
        coll_reward1 = [] 
        coll_reward2 = [] 

        loss_q1_episode = [] 
        loss_q2_episode = []  

        loss_w1_episode = [] 
        loss_w2_episode = [] 

        pol1_sel_episode = [] 
        pol2_sel_episode = []  

        avg_response_time = []
        costs = []
        chosen_confs = []
        chosen_confs_indexes = []
        requestCounter = [] 

        for episode in range(EPISODES):
            
            done = False
            starting_time = time.time()

            rewardsSum1 = 0
            rewardsSum2 = 0 

            qSum = 0

            qActions = 1

            lossSum = 0 

            num_steps = 0
            selected_policies = []
            

            nom_action, sel_policy, nominated_actions = agent.get_action_nomination(state)
            num_steps += 1 
            print('action selected : ', nom_action, 'policy selected : ', sel_policy)
            new_configuration_i = knowledge[nom_action] 
            eRI.change_configuration(configurations[nom_action])
            chosen_confs_indexes.append(nom_action)
            chosen_confs.append(configurations[nom_action])
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
             

            done = False   

            #sample_list = np.random.choice(sample_list, size = DESIRED_SAMPLES) #limit to 10 in case of over-sampling 

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

            reward = [-avg_response_time_of_episode, -cost_of_episode]

            selected_policies.append(sel_policy) 

            nextState = nextState 

            agent.store_transition(state, nom_action,reward, nextState, done, sel_policy)

            agent.learn()

            rewardsSum1 = np.add(rewardsSum1, reward[0]) 
            rewardsSum2 = np.add(rewardsSum2, reward[1]) 
            
            requestCounter.append(reqCount)
            reqCount=0  

            state = nextState
            done=True
            
            q_loss, w_loss = agent.get_loss_values()

            print("Episode", episode, "end_reward", reward, "Sum of the reward:", rewardsSum1+rewardsSum2, "Num steps:", num_steps,
                "Epsilon:", agent.epsilon, "Q loss:", q_loss, "W loss", w_loss, "request count : ", reqCount,)
            count_policies = collections.Counter(selected_policies)
            print("Policies selected in the episode:", count_policies, "Policy 1:", count_policies[0],
                "Policy 2:", count_policies[1], )
            count_policies = collections.Counter(selected_policies)
            
            # Save the performance to lists
            num_of_steps.append(num_steps)

            coll_reward1.append(rewardsSum1) 
            coll_reward2.append(rewardsSum2)  

            pol1_sel_episode.append(count_policies[0]) 
            pol2_sel_episode.append(count_policies[1])  

            loss_q1_episode.append(q_loss[0]) 
            loss_q2_episode.append(q_loss[1]) 

            loss_w1_episode.append(w_loss[0]) 
            loss_w2_episode.append(w_loss[1]) 

            avg_response_time.append(avg_response_time_of_episode)
            costs.append(cost_of_episode)

            ratioOfPolicy0= pol1_sel_episode.count(1)/len(pol1_sel_episode)
            ratioOfPolicy1= pol2_sel_episode.count(1)/len(pol2_sel_episode) 

            agent.update_params()
            
            print(f"time running : {time.time() - starting_time} s \n\n end of episode \n")

        # Save the results
        df_results = pd.DataFrame()
        df_results['episodes'] = range(1, EPISODES + 1) 
        df_results["chosen_conf_index"] = chosen_confs_indexes
        df_results["chosen_conf"] = chosen_confs
        df_results["chosen_conf_index_in_labels"] = df_results["chosen_conf"].apply(lambda x: labels.index([x,cost_dict[x]]))
        df_results["number_of_requests"] = requestCounter

        df_results['policy1'] = pol1_sel_episode 
        df_results['policy2'] = pol2_sel_episode 

        df_results['loss_q1'] = loss_q1_episode 
        df_results['loss_q2'] = loss_q2_episode  

        df_results['loss_w1'] = loss_w1_episode 
        df_results['loss_w2'] = loss_w2_episode 

        df_results['avg_response_time'] = avg_response_time
        df_results['cost'] = costs
         

        with open(f"./results/dwn/labels_dwn_{repeat}.pkl", 'wb+') as f:
            labels = [[confs_static.index(label[0]), label[1]] for label in labels]
            pickle.dump(labels, f)
        f.close()
            
        df_results.to_csv(f"./results/dwn/ews_dwn_{repeat}.csv")
        # Save the trained ANN
        agent.save(PATH+f"_{repeat}_") 







