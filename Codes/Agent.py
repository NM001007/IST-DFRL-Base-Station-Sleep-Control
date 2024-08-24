from PDQN_Network import *
from TP_Network import *
import torch

class Agent():
    def __init__(self, node_id, Train_data, Test_data, TP_hiddensize=100, TP_layers=2, RL_state=6, RL_action=1, 
                 explore_coefficient=0.2, decreasing_coefficient=0.2):
        self.ID = node_id
        self.Xtrain , self.Ytrain = Train_data
        self.Xtest, self.Ytest = Test_data

        self.current_mode = 1 # indicates the active/sleep mode of the base station: active=1; sleep=0

        self.TP_model = BiLSTMAttention(input_size=self.Xtrain.shape[2], tcn_input_size=self.Xtrain.shape[1],  
                                        hidden_size=TP_hiddensize, num_layers=TP_layers, output_size=1)
        self.DRL_model = DoubleDQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions = RL_action,
                                        eps_end=0.01, input_dims=RL_state, lr=0.0003)
        self.DRL_model.Q_eval.exploration_proba = 0.5

        self.Explore_model = DoubleDQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions = RL_action,
                                        eps_end=0.01, input_dims=RL_state, lr=0.0003)
        self.explore_initialize()
        self.explorer_coefficient = explore_coefficient
        self.decreasing_factor = decreasing_coefficient

        self.explorer_coefficient_decay = 0.005

        ## Cost parameters
        ## Energy Consumption
        self.fixed_energy_cost = 1.60 # 160 W => We have scaled it by 100 as the values of TP funciton are normally in range of (0,1)
        self.load_energy_ratio = 2.16 # 216 W ## The energy weight based on the current traffic load
        self.fixed_sleep_energy_cost = 0.24 ##ETRI journal

        self.transmission_cost = 0.20 # 20 s ## We assume a constant transmission cost between Base Stations
        # self.penalty_factor = 0.50 # 50 W/s ## for QoS degeration
        self.dynamic_penalty_weight = 3 #0.3 ## 3 seems the most efficient with the lower bound of 0.2 traffic load based on numerical analysis

        self.current_service_capacity = 1.00 ## We assume that the current service capacity resets every 30 minutes
        self.switching_cost = 1.00 # 100 Wh/time 


    def TP_train(self, batch_size, epochs):
        self.TP_model.learn(self, self.Xtrain, self.Ytrain, batch_size, epochs)
    
    def TP_evaluate(self):
        res = self.TP_model.forward(self.Xtest)
        evaluate_loss = self.TP_model.loss_fn(res, self.Ytest)
        return evaluate_loss

    def DL_train(self):
        self.DRL_model.learn()

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
        return penalty_factor*(self.transmission_cost + 1/(self.current_service_capacity-traffic_load))
    
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

    def explore_initialize(self):
        DRL_state =  self.DRL_model.Q_eval.state_dict()
        self.Explore_model.Q_eval.load_state_dict(DRL_state)

    def explore_update(self):
        rand_num = (random.random() * 2) - 1
        delta_W = self.Explore_model.Q_eval.state_dict()
        delta_W_coefficient = self.explorer_coefficient * rand_num
        for layer in delta_W:
            delta_W[layer] = delta_W_coefficient * delta_W[layer]
        
        Qval_W_noised = self.DRL_model.Q_eval.state_dict()
        for layer in Qval_W_noised:
            Qval_W_noised[layer] = Qval_W_noised[layer] + delta_W[layer]

        self.Explore_model.Q_eval.load_state_dict(Qval_W_noised)

    def update_actor(self):
        Explore_W = self.Explore_model.Q_eval.state_dict()
        Qval_W = self.DRL_model.Q_eval.state_dict()
        for layer in Qval_W:
            Qval_W[layer] = Qval_W[layer] + self.decreasing_factor * Explore_W[layer]
        
        self.DRL_model.Q_eval.load_state_dict(Qval_W)

    def update_explore_coefficient(self):
        self.explorer_coefficient = self.explorer_coefficient * np.exp(-self.explorer_coefficient_decay)
