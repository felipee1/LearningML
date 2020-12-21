import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

df=pd.read_csv("BBDC3.csv")
cps=df.shape
print('Number of rows and columns:', cps, "testante",int(cps[0]*0.1))
print(df.head(5))

training_set = df.iloc[:int(cps[0]-21), 1:2].values
test_set = df.iloc[1:int(cps[0]-21), 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(2,int(cps[0]-21)):
    X_train.append(training_set_scaled[i-2:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#(740, 60, 1)

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))


# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 16)

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:int(cps[0]-21), 1:2]
dataset_test = df.iloc[int(cps[0]/2):int(cps[0]-21), 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-2+21:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
print(inputs.shape)
for i in range(2, inputs.shape[0]):
    X_test.append(inputs[i-2:i, 0])
X_test = np.array(X_test)
print(X_test.shape[0], X_test.shape)
print(int(cps[0]-21))
#input()
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# (459, 60, 1)

predicted_stock_price = model.predict(X_test)
print(predicted_stock_price)
input()
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(df.loc[int(cps[0]/2):, "Date"],dataset_test.values, color = "red", label = 'Real Bradescão Stock Price')
plt.plot(df.loc[int(cps[0]/2):, 'Date'],predicted_stock_price, color = "blue", label = 'Predicted Bradescão Stock Price')
plt.xticks(np.arange(0,X_test.shape[0],20))
plt.title('BRADESCACHORRO Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('BRADESCACHORRO Stock Price')
plt.legend()
plt.show()