import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH = "./plots/gen/"
NUM_OF_SETS = 6  # Define the number of datasets to be loaded

def load_datasets(base_path, num_of_sets):
    data_dict = {}
    for i in range(num_of_sets):
        file_path = f"{base_path}{i}.csv"
        data_dict[i] = pd.read_csv(file_path)
    return data_dict   

def plot_multiple_shaded_data(data_dicts, metrics, titles, ylabel, filename, jump=10, fontsize=20, plot_title=''):

    plt.figure(figsize=(10, 5))

    plt.rcParams.update({'font.size': fontsize})
    
    colors = ['teal', 'purple', 'darkorange', 'gold', 'crimson', 'forestgreen']
    
    for idx, (data_dict, metric, title) in enumerate(zip(data_dicts, metrics, titles)):
        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', metric]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[idx % len(colors)]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'{title}', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

    plt.subplots_adjust(top=.9, bottom=.1, right=.9, left=.1, hspace=0, wspace=0) 
    plt.xlim([0, 600]) 

    font = {'family' : 'arial', 
        'size'   : 18}

    plt.rc('font', **font)

    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_multiple_policies_shaded_data(data_dicts, metrics, titles, ylabel, filename, jump=10, fontsize=20, plot_title=''):

    plt.figure(figsize=(10, 5))

    plt.rcParams.update({'font.size': fontsize})
    
    colors = ['teal', 'purple', 'gold', 'crimson', 'forestgreen', 'darkorange']
    
    for idx, (data_dict, metric, title) in enumerate(zip(data_dicts, metrics, titles)):
        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', metric]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[idx % len(colors)]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'{title} pol 0', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', f"not_{metric}"]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[len(colors)-(idx % len(colors))-1]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'{title} Pol 1', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)




    plt.subplots_adjust(top=.9, bottom=.1, right=.9, left=.1, hspace=0, wspace=0) 
    plt.xlim([0, 600]) 

    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_multiple_policies_shaded_plus_dwn_data(data_dicts, metrics, titles, ylabel, filename, jump=10, fontsize=20, plot_title=''):
    
    plt.figure(figsize=(10, 5))

    plt.rcParams.update({'font.size': fontsize})
    
    colors = ['teal', 'purple', 'gold', 'crimson', 'forestgreen', 'darkorange']
    
    for idx, (data_dict, metric, title) in enumerate(zip(data_dicts[0], metrics, titles)):
        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', metric]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[idx % len(colors)]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'DQN Time-Only Policy', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', f"not_{metric}"]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[len(colors)-(idx % len(colors))-1]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'DQN Cost-Only Policy', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
    
    for idx, (data_dict, metric, title) in enumerate(zip(data_dicts[1], metrics, titles)):
        # Combine all datasets into a single DataFrame for processing
        combined_df = pd.concat([df[['episodes', metric]].set_index('episodes') for df in data_dict.values()], axis=1)
        
        # Ensure unique indices
        combined_df = combined_df.reset_index().drop_duplicates(subset='episodes').set_index('episodes')
        
        # Resample to make jumps from 10 to 10
        combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
        
        # Calculate mean and standard deviation across all datasets
        mean_values = combined_df.mean(axis=1)
        std_values = combined_df.std(axis=1)
        
        color = colors[1]
        
        # Plot the mean line
        plt.plot(mean_values.index, mean_values, label=f'{title} Both Policies', color=color)
        
        # Fill the area between mean - std and mean + std
        plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

    plt.subplots_adjust(top=.9, bottom=.1, right=.9, left=.1, hspace=0, wspace=0) 
    plt.xlim([0, 600]) 

    font = {'family' : 'arial', 
        'size'   : 18}

    plt.rc('font', **font)

    plt.subplots_adjust(top=.9, bottom=.1, right=.9, left=.1, hspace=0, wspace=0) 
    plt.xlim([0, 600]) 

    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# DQN
base_path_requests_dqn = "./results/dqn/ews_dqn_"
data_dict_requests_dqn = load_datasets(base_path_requests_dqn, NUM_OF_SETS) 

# Egreedy
base_path_requests_egreedy = "./results/egreedy/ews_egreedy_"
data_dict_requests_egreedy = load_datasets(base_path_requests_egreedy, NUM_OF_SETS) 

# DWN
base_path_requests_dwn = "./results/dwn/ews_dwn_" 
data_dict_requests_dwn = load_datasets(base_path_requests_dwn, NUM_OF_SETS) 

#DWN with both policies
base_path_requests_dwn_both_pols = "./results/dwn/ews_dwn_both_pols" 
data_dict_requests_dwn_both_pols = load_datasets(base_path_requests_dwn_both_pols, NUM_OF_SETS) 

# DQN only optimizing average time
#base_path_requests_dqn_only_avg = "./results/dqn/ews_dqn_avg_"
#data_dict_requests_dqn_only_avg = load_datasets(base_path_requests_dqn_only_avg, NUM_OF_SETS) 

# DQN only optimizing peak time
#base_path_requests_dqn_only_peak = "./results/dqn/ews_dqn_peak_"
#data_dict_requests_dqn_only_peak = load_datasets(base_path_requests_dqn_only_peak, NUM_OF_SETS) 

# DQN only optimizing average and peak with weights
#base_path_requests_dqn_with_weights = "./results/dqn/ews_dqn_mix_"
#data_dict_requests_dqn_with_weights = load_datasets(base_path_requests_dqn_with_weights, NUM_OF_SETS)  


#All plots
#data_dicts = [data_dict_requests_dqn,data_dict_requests_dwn,data_dict_requests_egreedy,data_dict_requests_dqn_only_avg, data_dict_requests_dqn_only_peak, data_dict_requests_dqn_with_weights] 
#metrics_avgTime = ['avg_response_time', 'avg_response_time', 'avg_response_time','avg_response_time', 'avg_response_time', 'avg_response_time']
#titles_avgTime = ['DQN', 'DWN', 'E-greedy', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#metrics_peakTime = ['peak_time', 'peak_time', 'peak_time','peak_time', 'peak_time', 'peak_time']
#titles_peakTime = ['DQN', 'DWN', 'E-greedy', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#ylabel = 'Time (s)' 
#plot_multiple_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel, 'avg_response_time_plot_shaded_combined.png') 
#plot_multiple_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel, 'peak_time_plot_shaded_combined.png')

# DQN v DWN v E-greedy plots
data_dicts = [data_dict_requests_dqn,data_dict_requests_dwn,data_dict_requests_egreedy,]
metrics_avgTime = ['avg_response_time', 'avg_response_time', 'avg_response_time',]
titles_avgTime = ['DQN', 'DWN', '$\epsilon$-greedy',] 
ylabelAvg = 'Time (s)'
metrics_peakTime = ['cost', 'cost','cost']
titles_peakTime = ['DQN', 'DWN', '$\epsilon$-greedy', ] 
ylabelCost = 'Cost' 
plot_multiple_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabelAvg, 'avg_response_time_plot_shaded_combined.png', plot_title='Average Time Across Episodes') 
plot_multiple_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabelCost, 'cost_plot_shaded_combined.png', plot_title='Cost Across Episodes')


#DWN v DQN plots
#data_dicts = [data_dict_requests_dwn,data_dict_requests_dqn_only_avg, data_dict_requests_dqn_only_peak, data_dict_requests_dqn_with_weights] 
#metrics_avgTime = ['avg_response_time','avg_response_time', 'avg_response_time', 'avg_response_time']
#titles_avgTime = ['DWN', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#metrics_peakTime = ['peak_time','peak_time', 'peak_time', 'peak_time']
#titles_peakTime = ['DWN', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#ylabel = 'Time (s)' 
#plot_multiple_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel, 'average_DWN_comparison.png') 
#plot_multiple_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel, 'peak_DWN_comparison.png')


# DQN plots
#data_dicts = [data_dict_requests_dqn,data_dict_requests_dqn_only_avg, data_dict_requests_dqn_only_peak, data_dict_requests_dqn_with_weights] 
#metrics_avgTime = ['avg_response_time','avg_response_time', 'avg_response_time', 'avg_response_time']
#titles_avgTime = ['DQN', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#metrics_peakTime = ['peak_time','peak_time', 'peak_time', 'peak_time']
#titles_peakTime = ['DQN', 'DQN only avg', 'DQN only Peak', 'DQN with weights'] 
#ylabel = 'Time (s)' 
#plot_multiple_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel, 'average_DQNs.png') 
#plot_multiple_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel, 'peak_DQNs.png')


# only DWN
data_dicts = [data_dict_requests_dwn, ] 
metrics_avgTime = ['avg_response_time', ]
titles_avgTime = ['avg time', ] 
ylabel1 = 'Time (s)'
metrics_peakTime = ['cost', ]
titles_peakTime = ['Cost', ] 
ylabel2 = 'Cost' 
plot_multiple_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel1, 'DWN_avg.png',plot_title='Average Time Across Episodes') 
plot_multiple_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel2, 'DWN_cost.png',plot_title='Cost Across Episodes')


# separate policies of DWN
data_dicts = [data_dict_requests_dwn_both_pols ] 
metrics_avgTime = ['avg_response_time', ]
titles_avgTime = ['DWN', ] 
ylabel1 = 'Time (s)'
metrics_peakTime = ['cost', ]
titles_peakTime = ['DWN', ] 
ylabel2 = 'Cost' 
plot_multiple_policies_shaded_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel1, 'DWN_avg_both_pols.png',plot_title='Average Time Across Episodes') 
plot_multiple_policies_shaded_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel2, 'DWN_cost_both_pols.png',plot_title='Cost Across Episodes')

# all policies of DWN
data_dicts = [[data_dict_requests_dwn_both_pols],[data_dict_requests_dwn]] 
metrics_avgTime = ['avg_response_time', ]
titles_avgTime = ['DWN', ] 
ylabel1 = 'Time (s)'
metrics_peakTime = ['cost', ]
titles_peakTime = ['DWN', ] 
ylabel2 = 'Cost' 
plot_multiple_policies_shaded_plus_dwn_data(data_dicts, metrics_avgTime, titles_avgTime, ylabel1, 'DWN_avg_both_pols_plus_dwn.png',plot_title='Average Time Across Episodes') 
plot_multiple_policies_shaded_plus_dwn_data(data_dicts, metrics_peakTime, titles_peakTime, ylabel2, 'DWN_cost_both_pols_plus_dwn.png',plot_title='Cost Across Episodes')
