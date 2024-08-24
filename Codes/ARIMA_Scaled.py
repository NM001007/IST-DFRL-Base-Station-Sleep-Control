import pandas as pd
from utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

dataset_id = 0
Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

if __name__ == "__main__":
    #############################################
    ## evaluating total results

    data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
                 'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]
    
    data_df = read_from_pickle(data_path)
    train_test_ratio = 0.6

    folder_path = "ARIMA_10_20240807-202915"
    data_dfs = dict()
    for node in range(1,11):
        file_name = f"ARIMA_Node_{node}.csv"
        df = pd.read_csv(f"{folder_path}/{file_name}", usecols=['y_pred','y_true'])
        data_dfs[node] = df


    scalers = dict()
    scaler = MinMaxScaler(feature_range=(0, 1))
    Base_stations = dict()
    for node in range(1,11): 
        print(f"========= Node {node} ==========")
        test_pd = pd.DataFrame(data_dfs[node]['y_true'])
        pred_pd = pd.DataFrame(data_dfs[node]['y_pred']) 

        test_pd = test_pd.rename(columns={"y_true": "y"})
        pred_pd = pred_pd.rename(columns={"y_pred": "y"})

        test_pd = pd.DataFrame(scaler.fit_transform(test_pd))
        pred_pd = pd.DataFrame(scaler.transform(pred_pd))

        # evaluate forecasts
        MSE = mean_squared_error(test_pd, pred_pd)
        rmse = sqrt(mean_squared_error(test_pd, pred_pd))
        MAE = mean_absolute_error(test_pd, pred_pd)
        print('Test MSE: %.3f' % MSE)
        print('Test RMSE: %.3f' % rmse)
        print('Test MAE: %.3f' % MAE)
    





