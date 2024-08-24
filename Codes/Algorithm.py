from Agent import *
from PDQN_Network import *
from TP_Network import *
from utils import *

# Evaluation Function
def evaluate_models(Base_stations, train=True, eval=False):
    losses = []
    for model in Base_stations:
            Current_BS = Base_stations[model]
            if train:
                Xval = Current_BS.Xtrain
                Yval = Current_BS.Ytrain
            elif eval:
                Xval = Current_BS.Xtest
                Yval = Current_BS.Ytest
            Xval = torch.tensor(Xval)
            Yval = torch.tensor(Yval).reshape(Yval.shape[0], 1).detach().numpy()
            pred = Current_BS.TP_model(Xval).detach().numpy()

            t_ = torch.tensor(Yval - pred)
            loss = torch.mean(torch.square(t_))
            losses.append(loss.numpy())
    return losses


def FRL(feature_size, BS_datasets, num_nodes, RL_episodes, TP_Tepochs, TP_batchsize, time_slot_window, time_slot_steps,  
        DQN_update_frequency, DQN_target_frequency, explore_coefficient, decreasing_coefficient, 
        save_log, folder_path, from_checkpoint=False, checkpoint_path=None, episode_start=0):

    state_dim = 2  ## traffic volume, predicted active/sleep mode
    action_dim = 2 ## Only active/sleep modes; 0 indicates sleep and 1 indicates active 
    dim = feature_size+2 ## tensor dimensionality

    bs_keys = list(BS_datasets.keys())[0:num_nodes]
    print("Selected Base_Stations:", bs_keys)

    train_time_steps = 0
    eval_time_steps = 0
    Base_stations = dict()
    for node in bs_keys: 
        BS1 = BS_datasets[node]
        
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(BS1[0], BS1[1])
        train_time_steps = len(Xtrain)
        eval_time_steps = len(Xtest)

        Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
        Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))    

        Base_stations[node] = Agent(node, Train_data=(Xtrain, Ytrain), Test_data=(Xtest,Ytest),
                            TP_hiddensize=100, TP_layers=2, RL_state=state_dim, RL_action=action_dim,
                            explore_coefficient=explore_coefficient, decreasing_coefficient=decreasing_coefficient)

        if from_checkpoint:
            drl_model_name = f"BS{node}_DRL_Ep{episode_start-1}.pht"
            tp_model_name = f"BS{node}_TP_Ep{episode_start-1}.pht"
            Base_stations[node].TP_model.load_state_dict(torch.load(f"{checkpoint_path}/{tp_model_name}"))
            Base_stations[node].TP_model.eval()

            Base_stations[node].DRL_model.Q_eval.load_state_dict(torch.load(f"{checkpoint_path}/{drl_model_name}"))
            Base_stations[node].DRL_model.Q_eval.eval()

    print(f"Training Time Steps: {train_time_steps}")
    print(f"Evaluation Time Steps: {eval_time_steps}")

    ## Training Phase
    for episode in tqdm(list(range(RL_episodes)[episode_start:]), desc="\nDFRL Episodes"):

        global_timesteps = 0 ## timesteps increase in 10 minutes time intervals

        timesteps_results = []
        Avg_BS_results = []

        current_states = dict()
        next_states = dict()
        for node in bs_keys: 
            current_states[node] = np.zeros(shape=(1, state_dim))[0]
            next_states[node] = np.zeros(shape=(1, state_dim))[0]

        while global_timesteps < train_time_steps:
            avg_bs_energy = []
            avg_bs_QoS = []
            avg_bs_switching = []

            bss_e_costs = dict()
            bss_qos_costs = dict()
            bss_sw_costs = dict()
            traffics = dict()
            predicted_modes = dict()

            for bs in bs_keys:
                Current_BS = Base_stations[bs]
                data = Current_BS.Xtrain
                if global_timesteps %3 == 0: ## We assume that the current service capacity resets every 30 minutes (3 intervals)
                    Current_BS.current_service_capacity = 1
                elif global_timesteps > 0:
                    Current_BS.current_service_capacity -= Current_BS.Ytrain[global_timesteps-1]
                t = torch.tensor(data[global_timesteps]).reshape(1, dim, 1) 

                ## Estimating the mode of the BS based on the predicted traffic volume
                traffic_volume = Current_BS.TP_model(t)
                current_state = current_states[bs]
                current_state[0] = traffic_volume

                rand_n = np.random.uniform(0,1)
                if  rand_n < Current_BS.DRL_model.Q_eval.exploration_proba:
                    print("Random")
                    predicted_mode = np.random.randint(0,2)
                    costs = Current_BS.total_cost(traffic_volume, predicted_mode, return_all=True)
                else:
                    predicted_mode_actor = Current_BS.DRL_model.choose_action(current_state)  ## the output of the model
                    Current_BS.explore_update()
                    predicted_mode_explore = Current_BS.Explore_model.choose_action(current_state)  ## the output of the model

                    actor_cost_list = Current_BS.total_cost(traffic_volume, predicted_mode_actor, return_all=True)
                    actor_cost = sum(actor_cost_list)
                    explore_cost_list = Current_BS.total_cost(traffic_volume, predicted_mode_explore, return_all=True)
                    explore_cost = sum(explore_cost_list)

                    if explore_cost < actor_cost:   
                        predicted_mode = predicted_mode_explore
                        Current_BS.update_actor()
                        costs = explore_cost_list
                        print(f"{global_timesteps} >>> Explore Network; Actor costs:{actor_cost}; Explore costs:{explore_cost}")
                    else:
                        print("Actor:", predicted_mode_actor)
                        predicted_mode = predicted_mode_actor
                        costs = actor_cost_list
                
                bss_e_costs[bs] = costs[0]
                bss_qos_costs[bs] = costs[1]
                bss_sw_costs[bs] = costs[2]
                predicted_modes[bs] = predicted_mode
                traffics[bs] = traffic_volume

                avg_bs_energy.append(costs[0])
                avg_bs_QoS.append(costs[1])
                avg_bs_switching.append(costs[2])

                ts_result_dict = {'TimeStep': global_timesteps,
                                'BS': bs,
                                # 'Traffic': traffics[bs].detach().numpy()[0][0],
                                'Traffic': traffics[bs],
                                'A_S': predicted_modes[bs],
                                'EnergyCost': costs[0],
                                'QoS': costs[1],
                                'Switching': costs[2],
                                }
            
                timesteps_results.append(ts_result_dict)

                if predicted_mode != Current_BS.current_mode:
                    print(f'{global_timesteps}>> Base Station {bs} Mode Changed from {Current_BS.current_mode} to {predicted_mode}')
                    Current_BS.mode_status_change()

            num_activeBS = 0
            for bs in bs_keys:
                if Base_stations[bs].current_mode == 1:
                    num_activeBS += 1

            for bs in bs_keys: 
                next_states[bs] = np.zeros(shape=(1, state_dim))[0]
                next_states[bs][0] = traffics[bs]
                next_states[bs][1] = predicted_modes[bs]
                
                current_states[bs] = next_states[bs]
            
            qos_global = sum(list(bss_qos_costs.values()))
            sw_global = sum(list(bss_sw_costs.values()))
            for bs in bs_keys:
                e = bss_e_costs[bs]
                qos = qos_global
                sw = sw_global
                cs = [e, qos, sw]
                cs_cost = sum(cs)/100
                print(f"BS {bs} Traffic {traffics[bs]} Costs:{cs_cost}")
                Current_BS = Base_stations[bs]
                ## Store Transition Arguments: state, action, reward, new_state, done
                Current_BS.DRL_model.store_transition(current_states[bs], predicted_modes[bs], cs_cost, next_states[bs], 0)

            print("Num Active BS:", num_activeBS)
            print("==================")

            avg_bs_result = {'TimeStep': global_timesteps,
                             'EnergyCost': np.mean(avg_bs_energy),
                             'QoS': np.mean(avg_bs_QoS),
                             'Switching': np.mean(avg_bs_switching),}

            Avg_BS_results.append(avg_bs_result)

            if global_timesteps %500 == 0:
                print(f">> Global Time Step: {global_timesteps}")

            if global_timesteps %DQN_update_frequency == 0 and global_timesteps > 0:
                for bs in bs_keys:
                    Base_stations[bs].DRL_model.learn()
                    Base_stations[bs].DRL_model.Q_eval.update_exploration_probability()

            if global_timesteps % DQN_target_frequency == 0:
                for bs in bs_keys:
                    Base_stations[bs].DRL_model.target_Q = copy.deepcopy(Base_stations[bs].DRL_model.Q_eval)
                    
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

                ## DQN models
                DRL_state_1 = Base_stations[0].DRL_model.Q_eval.state_dict()
                for layer in DRL_state_1:
                    for index, bs in enumerate(bs_keys[1:]):
                        DRL_state_2 = Base_stations[bs].DRL_model.Q_eval.state_dict()
                        DRL_state_1[layer] = DRL_state_1[layer] + DRL_state_2[layer]
                    DRL_state_1[layer] = DRL_state_1[layer]/len(bs_keys)

                for bs in bs_keys:
                    Base_stations[bs].DRL_model.Q_eval.load_state_dict(DRL_state_1)

            global_timesteps += time_slot_steps
            # print("=============================")
            if save_log:
                pd.DataFrame(Avg_BS_results).to_csv(f"{folder_path}/Average_TimeStep_Results_Ep{episode}.csv")
        
        if save_log:
            training_resuls_episode = pd.DataFrame(timesteps_results)
            training_resuls_episode.to_csv(f"{folder_path}/Training_Ep_{episode}.csv")


        for bs in bs_keys:
            Base_stations[bs].update_explore_coefficient()
            Base_stations[bs].DRL_model.Q_eval.update_exploration_probability()
            print(f"BS explore coefficient: {Base_stations[bs].explorer_coefficient}")
            Base_stations[bs].DRL_model.learn()
            Base_stations[bs].DRL_model.target_Q = copy.deepcopy(Base_stations[bs].DRL_model.Q_eval)
            if save_log:
                ## Saving TP models
                torch.save(Base_stations[bs].TP_model.state_dict(), f"{folder_path}/BS{bs}_TP_Ep{episode}.pht")
                ## Saving DQN Actor model
                torch.save(Base_stations[bs].DRL_model.Q_eval.state_dict(), f"{folder_path}/BS{bs}_DRL_Ep{episode}.pht")

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

        bss_e_costs = dict()
        bss_qos_costs = dict()
        bss_sw_costs = dict()
        traffics = dict()
        predicted_modes = dict()
        
        for bs in bs_keys:
            Current_BS = Base_stations[bs]
            data = Current_BS.Xtest
            if global_index %3 == 0: ## We assume that the current service capacity resets every 30 minutes (3 intervals)
                Current_BS.current_service_capacity = 1
            elif global_index > 0:
                Current_BS.current_service_capacity -= Current_BS.Ytest[global_index-1]
            t = torch.tensor(data[global_index]).reshape(1, dim, 1) 

            ## Estimating the mode of the BS based on the predicted traffic volume
            traffic_volume = Current_BS.TP_model(t)
            current_state = current_states[bs]
            current_state[0] = traffic_volume

            rand_n = np.random.uniform(0,1)
            if  rand_n < Current_BS.DRL_model.Q_eval.exploration_proba:
                print("Random")
                predicted_mode = np.random.randint(0,2)
                costs = Current_BS.total_cost(traffic_volume, predicted_mode, return_all=True)
            else:
                predicted_mode_actor = Current_BS.DRL_model.choose_action(current_state)  ## the output of the model
                Current_BS.explore_update()
                predicted_mode_explore = Current_BS.Explore_model.choose_action(current_state)  ## the output of the model

                actor_cost_list = Current_BS.total_cost(traffic_volume, predicted_mode_actor, return_all=True)
                actor_cost = sum(actor_cost_list)
                explore_cost_list = Current_BS.total_cost(traffic_volume, predicted_mode_explore, return_all=True)
                explore_cost = sum(explore_cost_list)

                if explore_cost < actor_cost:   
                    predicted_mode = predicted_mode_explore
                    Current_BS.update_actor()
                    costs = explore_cost_list
                    print(f"{global_timesteps} >>> Explore Network; Actor costs:{actor_cost}; Explore costs:{explore_cost}")
                else:
                    print("Actor:", predicted_mode_actor)
                    predicted_mode = predicted_mode_actor
                    costs = actor_cost_list
                

            bss_e_costs[bs] = costs[0]
            bss_qos_costs[bs] = costs[1]
            bss_sw_costs[bs] = costs[2]
            predicted_modes[bs] = predicted_mode
            traffics[bs] = traffic_volume


            avg_bs_energy.append(costs[0])
            avg_bs_QoS.append(costs[1])
            avg_bs_switching.append(costs[2])

            ts_result_dict = {'TimeStep': global_timesteps,
                            'BS': bs,
                            # 'Traffic': traffics[bs].detach().numpy()[0][0],
                            'Traffic': traffics[bs],
                            'A_S': predicted_modes[bs],
                            'EnergyCost': costs[0],
                            'QoS': costs[1],
                            'Switching': costs[2],
                            }
            timesteps_results.append(ts_result_dict)

            tmstp_cost = {'TimeStep':global_timesteps ,'BS':bs, 'Energy':float(costs[0]), 'QoS':float(costs[1]), 'Sw':float(costs[2])}
            timestep_costs[bs].append(tmstp_cost)

            if predicted_mode != Current_BS.current_mode:
                print(f'{global_timesteps}>> Base Station {bs} Mode Changed from {Current_BS.current_mode} to {predicted_mode}')
                Current_BS.mode_status_change()

        num_activeBS = 0
        for bs in bs_keys:
            if Base_stations[bs].current_mode == 1:
                num_activeBS += 1
            else:
                print(f"BS {bs} is off.")

        for bs in bs_keys: 
            next_states[bs] = np.zeros(shape=(1, state_dim))[0]
            next_states[bs][0] = traffics[bs]
            next_states[bs][1] = predicted_modes[bs]
            current_states[bs] = next_states[bs]
            
        
        qos_global = sum(list(bss_qos_costs.values()))
        sw_global = sum(list(bss_sw_costs.values()))
        for bs in bs_keys:
            e = bss_e_costs[bs]
            qos = qos_global
            sw = sw_global
            cs = [e, qos, sw]
            cs_cost = sum(cs)/100
            print(f"BS {bs} Traffic {traffics[bs]} Costs:{cs_cost}")
            Current_BS = Base_stations[bs]
            ## Store Transition Arguments: state, action, reward, new_state, done
            Current_BS.DRL_model.store_transition(current_states[bs], predicted_modes[bs], cs_cost, next_states[bs], 0)

        print("Num Active BS:", num_activeBS)
        print("==================")

        print(f"Time: {global_timesteps} and Number of Active BSs: {num_activeBS}")

        avg_bs_result = {'TimeStep': global_timesteps,
                    'EnergyCost': np.mean(avg_bs_energy),
                    'QoS': np.mean(avg_bs_QoS),
                    'Switching': np.mean(avg_bs_switching),
                    'Num_ActiveBS': num_activeBS}

        Avg_Val_results.append(avg_bs_result)

        if global_timesteps %500 == 0:
            print(f">> Global Time Step: {global_timesteps}")

        if global_timesteps %DQN_update_frequency == 0 and global_timesteps > 0:
            for bs in bs_keys:
                Base_stations[bs].DRL_model.learn()
                Base_stations[bs].DRL_model.Q_eval.update_exploration_probability()

        if global_timesteps % DQN_target_frequency == 0:
            for bs in bs_keys:
                Base_stations[bs].DRL_model.target_Q = copy.deepcopy(Base_stations[bs].DRL_model.Q_eval)
                
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

            ## DQN models
            DRL_state_1 = Base_stations[0].DRL_model.Q_eval.state_dict()
            for layer in DRL_state_1:
                for index, bs in enumerate(bs_keys[1:]):
                    DRL_state_2 = Base_stations[bs].DRL_model.Q_eval.state_dict()
                    DRL_state_1[layer] = DRL_state_1[layer] + DRL_state_2[layer]
                DRL_state_1[layer] = DRL_state_1[layer]/len(bs_keys)

            for bs in bs_keys:
                Base_stations[bs].DRL_model.Q_eval.load_state_dict(DRL_state_1)

        global_timesteps += time_slot_steps
        global_index += time_slot_steps
        # print("=============================")

    for bs in bs_keys:
        Base_stations[bs].update_explore_coefficient()
        Base_stations[bs].DRL_model.Q_eval.update_exploration_probability()
        Base_stations[bs].DRL_model.learn()
        Base_stations[bs].DRL_model.target_Q = copy.deepcopy(Base_stations[bs].DRL_model.Q_eval)
        if save_log:
            ## Saving TP models
            torch.save(Base_stations[bs].TP_model.state_dict(), f"{folder_path}/BS{bs}_TP_Eval.pht")
            ## Saving DQN Actor model
            torch.save(Base_stations[bs].DRL_model.Q_eval.state_dict(), f"{folder_path}/BS{bs}_DRL_Eval.pht")

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
