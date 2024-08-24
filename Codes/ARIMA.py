from utils import *
from matplotlib import pyplot

# from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import datetime

dataset_id = 0
Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

num_nodes = 10
time_slot = 10 # minutes
time_slot_window = 48*6 ## 48 hours is the length of our time window
save_log = True

start_col = 3 # 2 has remaine
end_col = 5 #num_nodes+1
resume_time = 0

train_test_ratio = 0.6

p = 25
d = 1
q = 0 

if __name__ == "__main__":

    data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
                 'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]
    
    data_df = read_from_pickle(data_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"ARIMA_{num_nodes}_{timestamp}"
    if save_log:
        os.makedirs(output_dir, exist_ok=True)

    scaler = MinMaxScaler(feature_range=(0, 1))


    results_test = dict()
    results_test_scaled = dict()
    Base_stations = dict()
    for col in list(data_df.columns)[start_col:end_col]: 
        print(f"======= Base Station {col} =======")
        BS_data = data_df[col].values
        size = int(len(BS_data) * train_test_ratio)
        train, test = BS_data[0:size], BS_data[size:len(BS_data)]
        history = [x for x in train]
        predictions = list()

        results_test[col] = []
        results_test_scaled[col] = []

        # walk-forward validation
        for t in range(len(test))[resume_time:]: ## The model should go on a window of the last 48 (time slot window) time slots
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            history = history[-time_slot_window:] ## Adjusting the data on the last time slot window
            print(f'{t}>> predicted=%f, expected=%f' % (yhat, obs))
            results_test[col].append({'t': t, 'y_pred':yhat, 'y_true':obs})

            results_df = pd.DataFrame(results_test[col])
            results_df.to_csv(f'{output_dir}/ARIMA_Node_{col}.csv')
        
        test_pd = pd.DataFrame(scaler.fit_transform(pd.DataFrame(test)))[0:len(predictions)]
        pred_pd = pd.DataFrame(scaler.transform(pd.DataFrame(predictions)))
        
        results_test_scaled[col] = pd.concat([pred_pd, test_pd], axis=1)

        # evaluate forecasts
        MSE = mean_squared_error(test_pd, pred_pd)
        rmse = sqrt(mean_squared_error(test_pd, pred_pd))
        MAE = mean_absolute_error(test_pd, pred_pd)
        print('Test MSE: %.3f' % MSE)
        print('Test RMSE: %.3f' % rmse)
        print('Test MAE: %.3f' % MAE)

        filename = os.path.join(output_dir, f"ARIMA_Node_{col}.txt")
        with open(filename, "a") as file:
            file.write('Test MSE: %.3f' % MSE)
            file.write('Test RMSE: %.3f' % rmse)
            file.write('Test MAE: %.3f' % MAE)
        file.close()

        # # plot forecasts against actual outcomes
        # pyplot.plot(test)
        # pyplot.plot(predictions, color='red')
        # pyplot.show()

        results_df = pd.DataFrame(results_test[col])
        results_df.to_csv(f'{output_dir}/ARIMA_Node_{col}.csv')

        results_df_scaled = pd.DataFrame(results_test_scaled[col])
        results_df_scaled.to_csv(f'{output_dir}/ARIMA_Scaled_Node_{col}.csv')
