import os
import subprocess

# Define the paths to your scripts
#script_paths = ["./ews_dwn.py"]  
script_paths = [ "./other_algorithms/Envelope_MORL/synthetic/train_copy.py","./ews_dwn.py", "./ews_dwn_both_policies.py", "./ews_dqn.py", "./ews_egreedy.py", ] 

# Name of the Conda environment
env_comand = ""  #"..\\.actEnv\\Scripts\\activate" you should put the route to the environment you created while installing requirements.txt

REPEATS = 1

count = 0 

while count < REPEATS:
    for script in script_paths:
        print(f"Running {script}...")
        # Activate the environment
        if env_comand != "" : 
            activate_cmd = f"{env_comand} && python {script}"
            process = subprocess.Popen(activate_cmd, shell=True)
            process.wait()  # Wait for the process to finish
            print("\n\n\n\n")
        else : 
            activate_cmd = f"python {script}"
            process = subprocess.Popen(activate_cmd, shell=True)
            process.wait()  # Wait for the process to finish
            print("\n\n\n\n")

    print("All scripts have finished running.")
    count += 1
