from utils import *
from TP_Network import *
from Algorithm import *
import os
from datetime import datetime
import torch

class Agent():
    def __init__(self, node_id, Train_data, Test_data, TP_hiddensize=100, TP_layers=2):
        self.ID = node_id
        self.Xtrain , self.Ytrain = Train_data
        self.Xtest, self.Ytest = Test_data

        self.current_mode = 1 # indicates the active/sleep mode of the base station: active=1; sleep=0

        self.TP_model = BiLSTMAttention(input_size=self.Xtrain.shape[2], tcn_input_size=self.Xtrain.shape[1],  
                                        hidden_size=TP_hiddensize, num_layers=TP_layers, output_size=1)

        self.explorer_coefficient_decay = 0.005

        ## Cost parameters
        ## Energy Consumption
        self.fixed_energy_cost = 1.60 # 160 W => We have scaled it by 100 as the values of TP funciton are normally in range of (0,1)
        self.load_energy_ratio = 2.16 # 216 W ## The energy weight based on the current traffic load
        self.fixed_sleep_energy_cost = 0.24 ##ETRI journal

        self.transmission_cost = 0.20 # 20 s ## We assume a constant transmission cost between Base Stations
        # self.penalty_factor = 0.50 # 50 W/s ## for QoS degeration
        self.dynamic_penalty_weight = 3 ## 3 seems the most efficient with the lower bound of 0.2 traffic load based on numerical analysis

        self.current_service_capacity = 1.00 ## We assume that the current service capacity resets every 30 minutes
        self.switching_cost = 1.00 # 100 Wh/time 


    ## Reference: Deep RL with Spatio-Temporal 
    ## Based on generalized energy model
    ## p^t_i = fixed_cost + load_based_cost = P^f_i + ro^t_i * P^l_i
    def energy_consumption(self, predicted_mode, traffic_load=None): 
        if predicted_mode == 1: ## Active/ON
            return self.fixed_energy_cost + self.load_energy_ratio * traffic_load
        elif predicted_mode == 0: ## Sleep/OFF
            return self.fixed_sleep_energy_cost
    
    def QOS_degeration_cost(self, traffic_load):
        # return self.penalty_factor*(self.transmission_cost + 1/(self.current_service_capacity-traffic_load))
        penalty_factor = self.dynamic_penalty_weight*traffic_load
        # return penalty_factor*(self.transmission_cost + 1/abs(self.current_service_capacity-traffic_load))
        return penalty_factor*(self.transmission_cost + abs(self.current_service_capacity-traffic_load))
    
    def BS_switching_cost(self):
        return self.switching_cost

    def total_cost(self, traffic_load, predicted_mode, return_all=False):
    
        energy_cost, Qos_cost, switching = 0, 0, 0

        if self.current_mode == 1 and predicted_mode == 0: 
            ## predicted mode = Sleep, current_mode = Active => BS is about to sleep >> cost = sleep energy + QoS_deg
            ## There is no switching cost as the cost for switching is considered 0 when deactivating the BS;
            ## On the other hand, turning the BS ON will require the switching cost
            energy_cost = self.energy_consumption(predicted_mode)
            Qos_cost = self.QOS_degeration_cost(traffic_load)

        elif self.current_mode == 1 and predicted_mode == 1:
            ## predicted mode = Active, current_mode = Active => BS continues to be active >> cost = energy consumption based on 
            # load and fixed energy cost; No switching cost and no QoS degredation cost = 0
            energy_cost = self.energy_consumption(predicted_mode, traffic_load)

        elif self.current_mode == 0 and predicted_mode == 0: 
            ## The situation is similar to the sleep mode; The BS continues to be OFF and there are some QoS degredation;
            # there is no switching cost, as the BS is sleep.
            energy_cost = self.energy_consumption(predicted_mode)
            Qos_cost = self.QOS_degeration_cost(traffic_load)
        
        elif self.current_mode == 0 and predicted_mode == 1:
            ## In this situation, the BS is sleep and is required to get activated again. 
            # Thus, it needs the switching cost. No need for the QoS degredation and the energy consumption is based on the active mode.
            energy_cost = self.energy_consumption(predicted_mode, traffic_load)
            switching = self.BS_switching_cost()

        energy_cost *= 0.6
        Qos_cost *= 0.3
        switching *= 0.1

        if not return_all:
            return float(sum([energy_cost, Qos_cost, switching]))
        else:
            return [float(energy_cost), float(Qos_cost), float(switching)]
    
    def mode_status_change(self):
        self.current_mode = 0 if self.current_mode==1 else 1

#####################################

import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

#####################################
def Q_learning(feature_size, BS_datasets, num_nodes, Q_value_bound, time_slot_steps, folder_path):
    state_dim = 1  ## traffic volume
    action_dim = 2 ## Only active/sleep modes; 0 indicates sleep and 1 indicates active 
    dim = feature_size+2 ## tensor dimensionality
    
    Q_classes = [float(item*Q_value_bound) for item in range(0,10)]

    #Initialize the Q-table to 0 for distinct classes of traffic volume
    Q_table = np.zeros((len(Q_classes), action_dim))

    print(f"Q table initialized:\n{Q_table}")

    #initialize the exploration probability to 1
    exploration_proba = 1
    #exploartion decreasing decay for exponential decreasing
    exploration_decreasing_decay = 0.001
    # minimum of exploration proba
    min_exploration_proba = 0.01
    #discounted factor
    gamma = 0.99
    #learning rate
    lr = 0.1

    bs_keys = list(BS_datasets.keys())[0:num_nodes]
    print("Selected Base_Stations:", bs_keys)

    train_time_steps = 0
    eval_time_steps = 0
    Base_stations = dict()
    for node in bs_keys: 
        BS1 = BS_datasets[node]
        
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(BS1[0], BS1[1])
        print(Xtrain.shape)
        print(Xtest.shape)
        train_time_steps = len(Xtrain)
        eval_time_steps = len(Xtest)

        Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
        Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))    

        Base_stations[node] = Agent(node, Train_data=(Xtrain, Ytrain), Test_data=(Xtest,Ytest), TP_hiddensize=100, TP_layers=2)
        
    
    ## Training Phase
    for episode in tqdm(list(range(RL_episodes)), desc="\nDFRL Episodes"):

        global_timesteps = 0 ## timesteps increase in 10 minutes time intervals

        timesteps_results = []
        Avg_BS_results = []

        current_states = dict()
        next_states = dict()
        for node in bs_keys: 
            current_states[node] = np.zeros(shape=(1, state_dim))[0]
            next_states[node] = np.zeros(shape=(1, state_dim))[0]

        while global_timesteps < train_time_steps: 
            for bs in bs_keys:
                Current_BS = Base_stations[bs]
                data = Current_BS.Xtrain
                if global_timesteps %3 == 0: ## We assume that the current service capacity resets every 30 minutes (3 intervals)
                    Current_BS.current_service_capacity = 1
                elif global_timesteps > 0:
                    Current_BS.current_service_capacity -= Current_BS.Ytrain[global_timesteps-1]

                traffic_volume = current_states[node]
                value, index = find_nearest(Q_classes, traffic_volume)

                if np.random.uniform(0,1) < exploration_proba:
                    predicted_mode = np.random.randint(0,2)
                else:
                    predicted_mode = np.argmin(Q_table[index,:])

                costs = Current_BS.total_cost(traffic_volume, predicted_mode, return_all=True)
                cost = torch.clip(sum(torch.tensor(costs)), -1, 1)

                t = torch.tensor(data[global_timesteps]).reshape(1, dim, 1) 
                next_traffic = Current_BS.TP_model(t).detach().numpy()[0]
                
                _, next_state = find_nearest(Q_classes, next_traffic)
                Q_table[index, predicted_mode] = (1-lr) * Q_table[index, predicted_mode] +lr*(cost + gamma*min(Q_table[next_state,:]))
                
                current_states[node] = next_traffic

                if predicted_mode != Current_BS.current_mode:
                    print(f'{global_timesteps}>> Base Station {bs} Mode Changed from {Current_BS.current_mode} to {predicted_mode}')
                    Current_BS.mode_status_change()

            if global_timesteps %500 == 0:
                print(f">> Global Time Step: {global_timesteps}")

            if global_timesteps %time_slot_window == 0 and global_timesteps > 0: 
                ## Time to train the models and share the weights
               window_index = global_timesteps // time_slot_window
               print(f"Learning Time >> Full {time_slot_window/6} hours has past => {window_index}")

                for bs in bs_keys:
                    Current_BS = Base_stations[bs]
                    Xdata = Current_BS.Xtrain[(window_index-1)*time_slot_window : window_index*time_slot_window]
                    Ydata = Current_BS.Ytrain[(window_index-1)*time_slot_window : window_index*time_slot_window]
                    Xdata = torch.tensor(Xdata)
                    Ydata = torch.tensor(Ydata).reshape(Ydata.shape[0], 1)

                    Current_BS.TP_model.learn(Xdata, Ydata, TP_batchsize, TP_Tepochs)
                
                TP_state_1 = Base_stations[0].TP_model.state_dict()
                for layer in TP_state_1:
                    for index, bs in enumerate(bs_keys[1:]):
                        TP_state_2 = Base_stations[bs].TP_model.state_dict()
                        TP_state_1[layer] = TP_state_1[layer] + TP_state_2[layer]
                    TP_state_1[layer] = TP_state_1[layer]/len(bs_keys)

                for bs in bs_keys:
                    Base_stations[bs].TP_model.load_state_dict(TP_state_1)

            global_timesteps += time_slot_steps
        # We update the exploration proba using exponential decay formula 
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*episode))


    #####################################################################
    ########################## Evaluation Phase #########################
    #####################################################################

    print("""#####################################################################
########################## Evaluation Phase #########################
#####################################################################""")
    

    ## Evaluation Phase
    timestep_costs = dict()
    for node in bs_keys: 
        timestep_costs[node] = []

    Avg_Val_results = []
    global_index = 0
    while global_index < eval_time_steps: 
        avg_bs_energy = []
        avg_bs_QoS = []
        avg_bs_switching = []
        for bs in bs_keys:
            Current_BS = Base_stations[bs]
            data = Current_BS.Xtest
            if global_index %3 == 0: ## We assume that the current service capacity resets every 30 minutes (3 intervals)
                Current_BS.current_service_capacity = 1
            elif global_index > 0:
                Current_BS.current_service_capacity -= Current_BS.Xtest[global_index-1]

            traffic_volume = current_states[node]
            value, index = find_nearest(Q_classes, traffic_volume)

            if np.random.uniform(0,1) < exploration_proba:
                predicted_mode = np.random.randint(0,2)
            else:
                predicted_mode = np.argmin(Q_table[index,:])

            costs = Current_BS.total_cost(traffic_volume, predicted_mode, return_all=True)
            cost = torch.clip(sum(torch.tensor(costs)), -1, 1)

            t = torch.tensor(data[global_index]).reshape(1, dim, 1) 
            next_traffic = Current_BS.TP_model(t).detach().numpy()[0]
            _, next_state = find_nearest(Q_classes, next_traffic)
            Q_table[index, predicted_mode] = (1-lr) * Q_table[index, predicted_mode] +lr*(cost + gamma*min(Q_table[next_state,:]))
            
            current_states[node] = next_traffic

            avg_bs_energy.append(costs[0])
            avg_bs_QoS.append(costs[1])
            avg_bs_switching.append(costs[2])

            validation_ts_result_dict = {'Day': t[0][0].detach().numpy()[0],
                            'Time': t[0][1].detach().numpy()[0],
                            'TimeStep': global_timesteps,
                            'BS': bs,
                            'Traffic': traffic_volume,
                            'A_S': predicted_mode,
                            'EnergyCost': costs[0],
                            'QoS': costs[1],
                            'Switching': costs[2],
                            'State': cost
                            }
            
            timesteps_results.append(validation_ts_result_dict)

            tmstp_cost = {'TimeStep':global_timesteps ,'BS':bs, 'Energy':float(costs[0]), 'QoS':float(costs[1]), 'Sw':float(costs[2])}
            timestep_costs[bs].append(tmstp_cost)

            if predicted_mode != Current_BS.current_mode:
                print(f'{global_timesteps}>> Base Station {bs} Mode Changed from {Current_BS.current_mode} to {predicted_mode}')
                Current_BS.mode_status_change()

        num_activeBS = 0
        for bs in bs_keys:
            if Base_stations[bs].current_mode == 1:
                num_activeBS += 1

        avg_bs_result = {'TimeStep': global_timesteps,
                    'EnergyCost': np.mean(avg_bs_energy),
                    'QoS': np.mean(avg_bs_QoS),
                    'Switching': np.mean(avg_bs_switching),
                    'Num_ActiveBS': num_activeBS}

        Avg_Val_results.append(avg_bs_result)


        if global_timesteps %500 == 0:
            print(f">> Global Time Step: {global_timesteps}")

        if global_index %time_slot_window == 0 and global_index > 0: 
            ## Time to train the models and share the weights
            window_index = global_index // time_slot_window
            print(f"Learning Time >> Full {time_slot_window/6} hours has past => {window_index}")

            for bs in bs_keys:
                Current_BS = Base_stations[bs]
                Xdata = Current_BS.Xtest[(window_index-1)*time_slot_window : window_index*time_slot_window]
                Ydata = Current_BS.Ytest[(window_index-1)*time_slot_window : window_index*time_slot_window]
                Xdata = torch.tensor(Xdata)
                Ydata = torch.tensor(Ydata).reshape(Ydata.shape[0], 1)

                Current_BS.TP_model.learn(Xdata, Ydata, TP_batchsize, TP_Tepochs)
            
            TP_state_1 = Base_stations[0].TP_model.state_dict()
            for layer in TP_state_1:
                for index, bs in enumerate(bs_keys[1:]):
                    TP_state_2 = Base_stations[bs].TP_model.state_dict()
                    TP_state_1[layer] = TP_state_1[layer] + TP_state_2[layer]
                TP_state_1[layer] = TP_state_1[layer]/len(bs_keys)

            for bs in bs_keys:
                Base_stations[bs].TP_model.load_state_dict(TP_state_1)

        global_timesteps += time_slot_steps
        global_index += time_slot_steps
    # We update the exploration proba using exponential decay formula 
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*episode))

    if save_log:
        validation_results = pd.DataFrame(timesteps_results)
        validation_results.to_csv(f"{folder_path}/Validation_results.csv")
        pd.DataFrame(Avg_Val_results).to_csv(f"{folder_path}/Average_TimeStep_Results_Validation.csv")
        

    ## Evaluation Results
    Bs_costs = []
    energy_avg, QoS_avg, SW_avg = [], [], []
    Normalized_energy_avg, Normalized_QoS_avg, Normalized_SW_avg = [], [], []

    ## Saving the configurations file
    if save_log:
        filename = os.path.join(folder_path, f"Validation_Results_Summary.txt")
        with open(filename, "a") as file:
            if save_log:
                file.write("\nValidation Results (Summary):")
            
            for bs in bs_keys:
                print(f"======= Base Station {bs} =======")
                t_costs_pd = pd.DataFrame(timestep_costs[bs])

                if save_log:         
                    file.write(f"\n======= Base Station {bs} =======")
                    t_costs_pd.to_csv(f'{folder_path}/Eval_BS {bs}_Costs.csv')
                mean_costs = torch.tensor(t_costs_pd.mean())[-3:]

                print(f"Average Cost values (Evaluation) >> {mean_costs}")
                Bs_costs.append(mean_costs)
                normalized_cost = torch.nn.functional.normalize(mean_costs, dim=0)
                print(f"Normalized Average Cost values (Evaluation) >> {normalized_cost}")
                
                if save_log:
                    file.write(f"\nAverage Cost values (Evaluation) >> {mean_costs}")
                    file.write(f"\nNormalized Average Cost values (Evaluation) >> {normalized_cost}")

                energy_avg.append(mean_costs[0])
                QoS_avg.append(mean_costs[1])
                SW_avg.append(mean_costs[2])

                Normalized_energy_avg.append(normalized_cost[0])
                Normalized_QoS_avg.append(normalized_cost[1])
                Normalized_SW_avg.append(normalized_cost[2])

            print("\n==============================\n")
            print(f"Energy Average: {np.mean(energy_avg)}")
            print(f"QoS Average: {np.mean(QoS_avg)}")
            print(f"Switching Average: {np.mean(SW_avg)}")

            print(f"Normalized Energy Average: {np.mean(Normalized_energy_avg)}")
            print(f"Normalized QoS Average: {np.mean(Normalized_QoS_avg)}")
            print(f"Normalized Switching Average: {np.mean(Normalized_SW_avg)}")

            if save_log:
                file.write("\n==============================\n")
                file.write(f"\nEnergy Average: {np.mean(energy_avg)}")
                file.write(f"\nQoS Average: {np.mean(QoS_avg)}")
                file.write(f"\nSwitching Average: {np.mean(SW_avg)}")

                file.write(f"\n\nNormalized Energy Average: {np.mean(Normalized_energy_avg)}")
                file.write(f"\nNormalized QoS Average: {np.mean(Normalized_QoS_avg)}")
                file.write(f"\nNormalized Switching Average: {np.mean(Normalized_SW_avg)}")

            file.close()


dataset_id = 0
Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

num_nodes = 10
time_slot_steps = 3 # * 10-minutes
time_slot_window = 48*6 ## 48 hours is the length of our time window
feature_size = 6 #48 or 6 or 3 ## Time Slot Window

save_log = True

from_checkpoint = False
checkpoint_path = "Q_5_20240807-090311"
episode_start = 0

if __name__ == "__main__":


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"QL_{num_nodes}_{timestamp}"
    if save_log:
        os.makedirs(output_dir, exist_ok=True)



    data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
                 'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]
    
    data_df = read_from_pickle(data_path)

    ## Data for Telecome Italia is collected from 22:00 10/31/2013 to 22:50 12/19/2013.
    data_df = dataset_preprocess(data_df, initial=132, save=False, file_name=Dataset)
    BS_datasets, scalar = create_dataset(data_df, feature_size, num_nodes)
    print(len(data_df))

    Q_value_bound = 0.1

    RL_episodes = 10
    TP_Tepochs = 3
    TP_batchsize = 32


    ## Saving the configurations file
    if save_log:
        filename = os.path.join(output_dir, f"Experiment_Config.txt")
        with open(filename, "a") as file:
            file.write("\nSettings:")
            file.write(f"\nBS stations: {num_nodes}\n")
            file.write(f"Time_slot_Step: {time_slot_steps}\n")
            file.write(f"Time_slot_window: {time_slot_window}\n")
            file.write(f"Feature_size: {feature_size}\n")

            file.write("\nConfig:")
            file.write(f"RL_Episodes: {RL_episodes}\n")
            file.write(f"TP_Tepochs: {TP_Tepochs}\n")
            file.write(f"TP_batchsize: {TP_batchsize}\n")
            file.write(f"Q_value_bound:{Q_value_bound}")

        file.close()


    Q_learning(feature_size, BS_datasets, num_nodes, Q_value_bound, time_slot_steps, folder_path=output_dir)
