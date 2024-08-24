import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from pytorch_tcn import TCN
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, cnn_input_channels, cnn_out_channels, cnn_stride, cnn_kernel_size, output_size, lr=0.001):
        super(CNN_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lr = lr

        self.in_channels = cnn_input_channels
        self.out_channels = cnn_out_channels
        self.stride = cnn_stride
        self.kernel_size = cnn_kernel_size

        self.output = output_size
        
        self.cnn = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        self.lstm1 = nn.LSTM(self.out_channels, hidden_size, num_layers, bidirectional=False, batch_first=True, dropout=0.2)
        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.bn_lstm = nn.BatchNorm1d(num_features=hidden_size)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1)

        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.MSELoss()  # binary cross entropy
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self,x):
        out = self.relu(self.cnn(x))
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.relu(self.bn_lstm(self.lstm1(out)[0]))

        out = self.sigmoid(self.fc(out))
        return out

    def call(self, x):
            return self.forward(x)

    def learn(self, Xdata, Ydata, batch_size, epochs):
        n_epochs = epochs    # number of epochs to run
        batch_size = batch_size  # size of each batch
        batches_per_epoch = len(Xdata) // batch_size

        avg_total_loss = []
        for epoch in tqdm(list(range(n_epochs)), desc="Epoch Training: "):
            # start_time = time.time()
            avg_loss = []
            for i in range(batches_per_epoch):
                start = i * batch_size
                # take a batch
                Xbatch = Xdata[start:start+batch_size]
                ybatch = Ydata[start:start+batch_size]
                # forward pass
                y_pred = self.forward(Xbatch)
                loss = self.loss_fn(y_pred, ybatch)
                avg_loss.append(loss.detach().numpy())
                avg_total_loss.append(loss.detach().numpy())
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()

            # finish_time = time.time()
            # print(f"Epoch {epoch}/{epochs}: {(finish_time-start_time):.2f}s;  Average_Epoch_Loss:{np.mean(avg_loss):.5f}, Total_Avg_Loss:{np.mean(avg_total_loss):.5f}")
        print(f"Total_Avg_Loss:{np.mean(avg_total_loss):.5f}")


from utils import train_test_split, read_from_pickle, dataset_preprocess
import copy, pandas as pd, os


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


    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_ = MinMaxScaler(feature_range=(0, 1))
    # traffic_data = pd.DataFrame(scaler.fit_transform(traffic_data))
    Time = scaler_.fit_transform(Time)
    Day = scaler_.fit_transform(Day)

    ## normalize the dataset
    scalers = dict()

    BS_datasets = dict()
    for col in list(traffic_data.columns)[0:n_nodes]:
        print(f"Colum {col} is being processed.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        traffic_col = pd.DataFrame(scaler.fit_transform(pd.DataFrame(traffic_data[col])))
        scalers[col] = scaler

        X, Y = [], []
        for i in range(len(traffic_col)-feature_size-1):
            a = traffic_col[0][i:(i+feature_size)]
            X.append(a)
            Y.append(traffic_col[0][i + feature_size])
        X = np.array(X)
        Y = np.array(Y)
        X = np.concatenate((Day, Time, X), axis=1)

        BS_datasets[col] = (X, Y)
        
    if return_df:
        return BS_datasets, scaler, pd.concat([Day_df, Time_df, traffic_data], axis=1)

    return BS_datasets, scalers


dataset_id = 0
Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

num_nodes = 10
feature_size = 3 #48 or 6 or 3 ## Time Slot Window

batch_size = 32
epochs = 10


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"CNN-LSTM_{num_nodes}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)



data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
                'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]

data_df = read_from_pickle(data_path)
## Data for Telecome Italia is collected from 22:00 10/31/2013 to 22:50 12/19/2013.
data_df = dataset_preprocess(data_df, initial=132, save=False, file_name=Dataset)
BS_datasets, scalar = create_dataset(data_df, feature_size, num_nodes)
print(len(data_df))

bs_keys = list(BS_datasets.keys())[0:num_nodes]

results_test = dict()
Base_stations = dict()
for node in bs_keys: 
    print(f"======= Base Station {node} =======")
    BS1 = BS_datasets[node]

    results_test[node] = []

    Xtrain, Ytrain, Xtest, Ytest = train_test_split(BS1[0], BS1[1])

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))   

    Xtrain = torch.tensor(Xtrain)
    Xtest = torch.tensor(Xtest)

    Ytrain = torch.tensor(Ytrain).reshape(Ytrain.shape[0], 1)

    cnn_lstm_model = CNN_LSTM(cnn_input_channels=Xtrain.shape[1], cnn_out_channels=64, cnn_stride=2, cnn_kernel_size=1,
                              hidden_size=100, num_layers=2, output_size=1)
    
    ## Training
    cnn_lstm_model.learn(Xtrain, Ytrain, batch_size, epochs)

    ## Evaluation
    pred = cnn_lstm_model(Xtest).detach().numpy()

    # evaluate forecasts
    MSE = mean_squared_error(Ytest, pred)
    rmse = sqrt(mean_squared_error(Ytest, pred))
    MAE = mean_absolute_error(Ytest, pred)
    print('Test MSE: %.3f' % MSE)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % MAE)

    for item in range(len(Ytest)):
        results_test[node].append({'y_pred':pred[item], 'y_true':Ytest[item]})

    results_df = pd.DataFrame(results_test[node])
    results_df.to_csv(f'{output_dir}/CNN-LSTM_Node_{node}.csv')

    filename = os.path.join(output_dir, f"CNN-LSTM_Node_{node}.txt")
    with open(filename, "a") as file:
        file.write('Test MSE: %.3f' % MSE)
        file.write('Test RMSE: %.3f' % rmse)
        file.write('Test MAE: %.3f' % MAE)
    file.close()