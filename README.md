# EWS Implementation Using DWN

This is the Repository for the article submitted to AI4AS 2024, Here we present the code for our imlementation of our EWS-DWN optimization algorithm, which we present in our article _Multi-Objective Deep Reinforcement Learning applied to Adaptive Web Servers_, by Ivana Dusparic, Nicolàs Cardozo, and Juan Rosero. A copy of the preprinted article is present in this repository.

## Index

* [Contents](##contents)
* [Instalation](##Instalation)
* [Execution](##Execution)
* [Credits](##Credits)

## Contents

this repository is organized in the following way, we have in the main folder the scripts for the execution of our DWN, DQN and ε-greedy algorithms, a script to run multiple scrips sequentially, and the requirements.txt file with the necessary python modules required to execute our code, bar from the ews module, which requires to be manually installed from the [PyEws repository](https://github.com/EGAlberts/pyews). then we have the following folders : 

* agents : contains the code for the RL agents we used in our implementations
* plots : contains the plots we created for our collected data, as well as the folder where our generate plot and tables scripts generate their respective content
* results : contains a .pkl file where we have the static costs associated with each configuration. this folder also contains several sub-folders where our result data is stored in separate csv files according to the number of the episode. we also have a .pkl file that stores the shuffled configurations used for their respective episode, it is formed by a touple array, for each configuration used, in the following manner `[index of the configuration in the list returned by the server, cost of the configuration]`
* utils : some auxiliar scripts we used to generate the plots and the tables. we recommend that if you are going to use one of these, you move it to the main folder, as it was there where it was used allways during our tests.

## Instalation

For windows : 
    
* Python 3.10+
* swigwin 4.2.1+
* mingw

The requirements.txt file contains the necessary python modules required to execute our code, bar from the ews module, which requires to be manually installed from the [PyEws repository](https://github.com/EGAlberts/pyews), so appart from EWS which, needs to be installed following the instructions present in their repository, the rest of the python requirements can be installed with a simple `pip install -r requirements.txt` from the main folder.

Aditionally, for the execution of EWS, we used a docker container available in the [EWS Repository](https://github.com/robertovrf/emergent_web_server).

## Execution

For the execution of any of the scripts containing our algorithms, EWS should already be running, accompanied by any of the client scripts created in EWS, the one used by us was `ClientTextPattern.o`

Once the client script is already running and the server is receiving requests, our scripts can be executed by either executing `run_ews_scripts.py` in which we can indicate which scripts we want to execute (by default it will execute all scripts) by modifying the `script_paths` array, if you installed our requirements in a local environment you can also modify `env_comand` to use it, by default the scripts uses the global python .exe. Our scripts can also be executed separatedly if that is the users preference.

Inside each of our scripts, there is the list of hyperparameters and configurations for the tests at the begining of each script, including the length and amount of episodes that we want to run.


## Credits

Ivana Dusparic (ivana.dusparic@tcd.ie)

Nicolàs Cardozo (n.cardozo@uniandes.edu.co)

Juan Camilo Rosero (roserolj@tcd.ie)
