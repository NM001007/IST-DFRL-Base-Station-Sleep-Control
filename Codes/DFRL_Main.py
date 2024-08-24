from utils import *
from Agent import *
from PDQN_Network import *
from TP_Network import *
from Algorithm import *
import os
from datetime import datetime

dataset_id = 0
Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

num_nodes = 10
time_slot_steps = 3 # minutes
time_slot_window = 48*6 ## 48 hours is the length of our time window
feature_size = 6 #48 or 6 or 3 ## Time Slot Window

save_log = True


from_checkpoint = False
checkpoint_path = ""
episode_start = 5
if not from_checkpoint:
    episode_start = 0

if __name__ == "__main__":


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"DFRL_{num_nodes}_{timestamp}"
    if save_log:
        os.makedirs(output_dir, exist_ok=True)

    data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
                 'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]
    
    data_df = read_from_pickle(data_path)

    ## Data for Telecome Italia is collected from 22:00 10/31/2013 to 22:50 12/19/2013.
    data_df = dataset_preprocess(data_df, initial=132, save=False, file_name=Dataset)
    BS_datasets, scalar = create_dataset(data_df, feature_size, num_nodes)
    print(len(data_df))

    RL_episodes = 10
    TP_Tepochs = 3
    TP_batchsize = 32
    DQN_update_frequency = 12*6
    DQN_target_frequency = 24*6
    explore_coefficient = 0.7
    decreasing_coefficient = 0.5

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
            file.write(f"DQN_update_frequency: {DQN_update_frequency}\n")
            file.write(f"DQN_target_frequency: {DQN_target_frequency}\n")
            file.write(f"explore_coefficient: {explore_coefficient}\n")
            file.write(f"decreasing_coefficient: {decreasing_coefficient}\n")
        file.close()

    FRL(feature_size, BS_datasets, num_nodes, RL_episodes, TP_Tepochs, TP_batchsize, time_slot_window, time_slot_steps,
        DQN_update_frequency, DQN_target_frequency, explore_coefficient, decreasing_coefficient, save_log,
        output_dir, 
        from_checkpoint=from_checkpoint, checkpoint_path=checkpoint_path, episode_start=episode_start)
