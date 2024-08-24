import pickle
import pandas as pd
import copy
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

## this function reads the instances from saved ones
def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]

    return pd.DataFrame(objects)


## Data for Telecome Italia is collected from 22:00 10/31/2013 to 22:50 12/19/2013.
## A day (24 hours) has 144 10-minutes intervals
## The time 22:00 can be indexed as the 132th 10-minutes interval in a day
## The time 22:50 can be indexed as the 137th 10-minutes interval in a day
## Similarly, each hour in a day can be regarded as hour*6 th 10-minutes interval
def dataset_preprocess(data, time_intervals=10, initial=0, save=False, file_path='./Datasets', file_name='TelecomItalia'): 
    data_df = copy.deepcopy(data)

    index = initial
    day_index = 0

    time_list = []
    day_list = []
    for _ in range(len(data_df)):
        if index == 144:
            index = 0 
            day_index = 0 if day_index == 6 else day_index+1
        else:
            index += 1
    
        day_list.append(day_index)
        time_list.append(index)

    times = pd.DataFrame(time_list, columns=['Time'])
    days = pd.DataFrame(day_list, columns=['Day'])
    data_df = pd.concat([days, times, data_df], axis=1)
    data_df = data_df.astype('float32')

    if save:
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        
        data_df.to_csv(f'{file_path}/{file_name}.csv')
    return data_df


def create_dataset(dataset:pd.DataFrame, feature_size:int, n_nodes, return_df=False):
    data_df = copy.deepcopy(dataset)
    traffic_data = data_df.drop(columns=['Time'])
    traffic_data = traffic_data.drop(columns=['Day'])

    Time_df = pd.DataFrame(data_df['Time'])
    Day_df = pd.DataFrame(data_df['Day'])

    Time = pd.DataFrame(data_df['Time'])
    Day = pd.DataFrame(data_df['Day'])
    Time.rename(columns={'Time': 0}, inplace=True)
    Day.rename(columns={'Day': -1}, inplace=True)

    Time = Time.shift(-feature_size-1).dropna()
    Day = Day.shift(-feature_size-1).dropna()

    ## normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_ = MinMaxScaler(feature_range=(0, 1))
    traffic_data = pd.DataFrame(scaler.fit_transform(traffic_data))
    Time = scaler_.fit_transform(Time)
    Day = scaler_.fit_transform(Day)

    BS_datasets = dict()
    for col in list(traffic_data.columns)[0:n_nodes]:
        print(f"Colum {col} is being processed.")
        X, Y = [], []
        for i in range(len(traffic_data[col])-feature_size-1):
            a = traffic_data[col][i:(i+feature_size)]
            X.append(a)
            Y.append(traffic_data[col][i + feature_size])
        X = np.array(X)
        Y = np.array(Y)
        X = np.concatenate((Day, Time, X), axis=1)

        BS_datasets[col] = (X, Y)
        
    if return_df:
        return BS_datasets, scaler, pd.concat([Day_df, Time_df, traffic_data], axis=1)

    return BS_datasets, scaler


def train_test_split(X, Y, train_rate=0.6):
    train_size = int(len(X) * train_rate)
    Xtrain, Xtest = X[0:train_size,:], X[train_size:len(X),:]
    Ytrain, Ytest = Y[0:train_size], Y[train_size:len(Y)]

    return Xtrain, Ytrain, Xtest, Ytest


