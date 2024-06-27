from pyews.server_interface import ewsRESTInterface as eRI 
from pyews.global_vars import settings
import pickle
import pandas as pd
import numpy as np
import time 


COLLECTION_WINDOW = 3
DESIRED_SAMPLES = 10
CHOSEN_METRIC = "response_time"
CONFIG = 0
REWARD = 1
N_K = 2
COST=3
EPSILON = 0.001 # 0.5
DATASET_SIZE = 600
REPEATS = 8


for repeat in range(REPEATS):
    definitions = {
        "URL" : "http://localhost:2011/",    
        "main_component" : "../repository/TCPNetwork.o",
        "proxy_JSON" : {"exp":"|../pal/monitoring/proxies/HTTPProxy.o|*(*:http.handler.GET.HTTPGET[0]:*)|"}
    } 

    settings["IP"] = definitions["URL"]

    #eRI.initialize_server(definitions["main_component"],definitions["proxy_JSON"])

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

    knowledge = [[config, 0, 1,cost_dict[config]] for config in configurations] #[config, cumulative reward, times_chosen,peak_time] 


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

    def e_greedy(arms, knowldg):
        
        choice = np.random.random()

        selected_arm = None
        if choice < EPSILON:  
            selected_arm = np.random.choice(len(arms))
        else:
            index = knowldg.index(min(knowldg, key=lambda k: (k[REWARD]/k[N_K])+k[COST])) #max of the averages, and then the index of that. 
            selected_arm = index

        return selected_arm

    ind=1

    avg_response_times = []
    costs = []
    chosen_confs = []
    chosen_confs_indexes = []
    requestCounter = []

    while(ind < DATASET_SIZE+1):
        new_configuration_i = e_greedy(configurations, knowledge)
        eRI.change_configuration(configurations[new_configuration_i])
        chosen_confs_indexes.append(new_configuration_i)
        chosen_confs.append(configurations[new_configuration_i])
        print(f"{ind} : new config created, config index : {new_configuration_i}")
        cost_of_episode = cost_dict[configurations[new_configuration_i]]
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

        costs.append(cost_of_episode)
        
        #sample_list = np.random.choice(sample_list, size = DESIRED_SAMPLES) #limit to 10 in case of over-sampling

        avg_response_time_of_episode = sum(sample_list)/reqCount

        if avg_response_time_of_episode > 0.5 :
                avg_response_time_of_episode = 2
                cost_of_episode = 2

        if cost_of_episode > 0.5 : 
            avg_response_time_of_episode = 2
            cost_of_episode = 2

        avg_response_times.append(avg_response_time_of_episode)
        
        print("\n\n\n\n configuration times : \n\n  avg response time : ", avg_response_time_of_episode, "cost : " , cost_of_episode, "number of requests : ", requestCounter, "\n\n\n\n")
 
        knowledge[new_configuration_i][N_K] += 1 
        knowledge[new_configuration_i][REWARD] += avg_response_time_of_episode  
        knowledge[new_configuration_i][COST] = cost_of_episode  

        requestCounter.append(reqCount)
        reqCount=0 

        ind = ind + 1

        # Save the results
    df_results = pd.DataFrame()
    df_results['episodes'] = range(1, DATASET_SIZE + 1) 
    df_results["chosen_conf_index"] = chosen_confs_indexes
    df_results["chosen_conf"] = chosen_confs
    df_results["chosen_conf_index_in_labels"] = df_results["chosen_conf"].apply(lambda x: labels.index([x,cost_dict[x]]))
    df_results["number_of_requests"] = requestCounter

    df_results['avg_response_time'] = avg_response_times
    df_results['cost'] = costs 

    with open(f"./results/egreedy/labels_egreedy_{repeat}.pkl", 'wb+') as f:
        labels = [[confs_static.index(label[0]), label[1]] for label in labels]
        pickle.dump(labels, f)
    f.close()

    df_results.to_csv(f"./results/egreedy/ews_egreedy_{repeat}.csv")






