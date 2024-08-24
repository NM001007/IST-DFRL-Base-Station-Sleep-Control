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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.feature_size = hidden_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(self.feature_size, self.feature_size)
        self.query = nn.Linear(self.feature_size, self.feature_size)
        self.value = nn.Linear(self.feature_size, self.feature_size)

    def forward(self, x):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, tcn_input_size, output_size, lr=0.001):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr

        self.do = nn.Dropout1d()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.relu = nn.ReLU()
        self.bn_lstm = nn.BatchNorm1d(num_features=tcn_input_size)

        self.tcn = TCN(num_inputs=tcn_input_size, num_channels=[1, 2, 4], dilations=[1, 2, 4], activation='relu')
        self.tcn = TCN(num_inputs=tcn_input_size, num_channels=[1, 2], dilations=[1, 2], activation='relu')

        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.bn_tcn= nn.BatchNorm1d(num_features=2)

        self.flatten = nn.Flatten()

        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.MSELoss()  # binary cross entropy
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        # x = self.do(x)
        lstm_out = self.relu(self.bn_lstm(self.lstm1(x)[0]))
        lstm_out = self.relu(self.bn_lstm(self.lstm2(lstm_out)[0]))
        
        tcn_output = self.bn_tcn(self.tcn(lstm_out))

        maxpool = self.maxpool(tcn_output)
        avgpool = self.avgpool(tcn_output)
        concat = torch.concatenate([avgpool, maxpool], dim=2)

        context = self.attention(concat)
        
        flat = self.flatten(context)
        out = self.sigmoid(self.fc(flat))
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



## To evaluate the model separately, please uncomment the following lines.
# from utils import * 

# dataset_id = 0
# Dataset = {0:'telecomItalia', 1:'OpNet'}[dataset_id]
# # num_nodes = {'telecomItalia': 225, 'OpNet': 120}[Dataset]

# num_nodes = 11
# time_slot = 10 # minutes
# time_slot_window = 48*6 ## 48 hours is the length of our time window
# feature_size = 6 #48 or 6 or 3 ## Time Slot Window

# batch_size = 32
# epochs = 10

# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# output_dir = f"LSTM-TCN-ATT_{num_nodes}_{timestamp}"
# os.makedirs(output_dir, exist_ok=True)


# data_path = {'telecomItalia': './Datasets/telecomItalia/telecomItalia.pkl',
#                 'OpNet': './Datasets/opnet/opnet.pkl'}[Dataset]

# data_df = read_from_pickle(data_path)
# ## Data for Telecome Italia is collected from 22:00 10/31/2013 to 22:50 12/19/2013.
# data_df = dataset_preprocess(data_df, initial=132, save=False, file_name=Dataset)
# BS_datasets, scalar = create_dataset(data_df, feature_size, num_nodes)
# print(len(data_df))

# bs_keys = list(BS_datasets.keys())[10:num_nodes]


# results_test = dict()
# Base_stations = dict()
# for node in bs_keys: 
#     print(f"======= Base Station {node} =======")
#     BS1 = BS_datasets[node]
#     results_test[node] = []
    
#     Xtrain, Ytrain, Xtest, Ytest = train_test_split(BS1[0], BS1[1])

#     Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
#     Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))   

#     Xtrain = torch.tensor(Xtrain)
#     Xtest = torch.tensor(Xtest)

#     Ytrain = torch.tensor(Ytrain).reshape(Ytrain.shape[0], 1)

#     bilstm_attention = BiLSTMAttention(input_size=Xtrain.shape[2], tcn_input_size=Xtrain.shape[1],  
#                                         hidden_size=100, num_layers=2, output_size=1)
    
#     bilstm_attention.learn(Xtrain, Ytrain, batch_size, epochs)

#     ## Evaluation
#     pred = bilstm_attention(Xtest).detach().numpy()

#     # evaluate forecasts
#     MSE = mean_squared_error(Ytest, pred)
#     rmse = sqrt(mean_squared_error(Ytest, pred))
#     MAE = mean_absolute_error(Ytest, pred)
#     print('Test MSE: %.3f' % MSE)
#     print('Test RMSE: %.3f' % rmse)
#     print('Test MAE: %.3f' % MAE)

#     for item in range(len(Ytest)):
#         results_test[node].append({'y_pred':pred[item], 'y_true':Ytest[item]})

#     results_df = pd.DataFrame(results_test[node])
#     results_df.to_csv(f'{output_dir}/LSTM-TCN-ATT_Node_{node}.csv')

#     filename = os.path.join(output_dir, f"LSTM-TCN-ATT_Node_{node}.txt")
#     with open(filename, "a") as file:
#         file.write('Test MSE: %.3f' % MSE)
#         file.write('Test RMSE: %.3f' % rmse)
#         file.write('Test MAE: %.3f' % MAE)
#     file.close()
