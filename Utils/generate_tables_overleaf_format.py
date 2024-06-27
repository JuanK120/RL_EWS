import pandas as pd
import numpy as np

PATH = "./plots/gen/"
NUM_OF_SETS = 8  # Define the number of datasets to be loaded

def load_datasets(base_path, num_of_sets):
    data_dict = {}
    for i in range(num_of_sets):
        file_path = f"{base_path}{i}.csv"
        data_dict[i] = pd.read_csv(file_path)
    return data_dict

def calculate_statistics(data_dict, metric, jump=10):
    # Combine all datasets into a single DataFrame for processing 
    combined_df = pd.concat([df[['episodes', metric]].set_index('episodes') for df in data_dict.values()], axis=1)
    
    # Resample to make jumps from 10 to 10
    combined_df = combined_df.groupby(combined_df.index // jump * jump).mean()
    
    # Select the last 100 steps
    last_hundred_steps = combined_df.tail(100 // jump)
    
    # Calculate mean and standard deviation across all datasets
    mean_values = last_hundred_steps.mean(axis=1)
    std_values = last_hundred_steps.std(axis=1)
    
    return mean_values, std_values

def write_statistics_to_file(data_dicts, metrics, titles, filename, jump=10):
    with open(PATH + filename, 'w') as file:
        # Write the LaTeX table header
        file.write("\\begin{tabular}{c || c|c|c|c|c|c}\n")
        file.write("\\hline\n")  

        for idx, (data_dict, metric, title) in enumerate(zip(data_dicts, metrics, titles)):
            mean_values, std_values = calculate_statistics(data_dict, metric, jump)
            
            file.write(f"\\textbf{{{title} - {metric.replace('_', ' ')}}} \\\\\n")
            for metric_type, values in zip([''], [mean_values, std_values]):
                file.write(f"\\textbf{{{metric_type}}} & ")
                file.write(" & ".join([f"{mean:.4f} $\\pm$ {std_dev:.4f}" for mean, std_dev in zip(mean_values, std_values)]))
                file.write(" \\\\\n")
            file.write("\\hline\n")
        file.write("\\end{tabular}\n")

# Load datasets
base_path_dqn = "./results/dqn/ews_dqn_"
base_path_egreedy = "./results/egreedy/ews_egreedy_"
base_path_dwn = "./results/dwn/ews_dwn_"
base_path_dwn_both_pols = "./results/dwn/ews_dwn_both_pols"

data_dict_dqn = load_datasets(base_path_dqn, NUM_OF_SETS)
data_dict_egreedy = load_datasets(base_path_egreedy, NUM_OF_SETS)
data_dict_dwn = load_datasets(base_path_dwn, NUM_OF_SETS)
data_dict_dwn_both_pols = load_datasets(base_path_dwn_both_pols, NUM_OF_SETS)

data_dicts = [data_dict_dwn, data_dict_dqn, data_dict_egreedy,data_dict_dwn_both_pols,data_dict_dwn_both_pols]

metrics = ['avg_response_time', 'avg_response_time', 'avg_response_time','avg_response_time','not_avg_response_time']
titles = ['DWN','DQN', 'Egreedy','DWN_pol1','DWN_pol2']

# Write statistics for avg_response_time to file
write_statistics_to_file(data_dicts, metrics, titles, 'avg_response_time_statistics.txt', jump=100)

metrics = ['cost', 'cost', 'cost','cost','not_cost']
titles = ['DWN','DQN', 'Egreedy','DWN_pol1','DWN_pol2']

# Write statistics for cost to file
write_statistics_to_file(data_dicts, metrics, titles, 'cost_statistics.txt', jump=100)
