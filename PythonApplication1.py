

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


def plot_predictions(test,predicted):
    plt.plot(test, color='red', label='Real IBM Stock Price')
    plt.plot(predicted, color='blue', label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))



dataset = pd.read_csv(r"C:\Users\simpl\Machine Learning\PythonApplication1\IBM_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
dataset.head()
print(dataset)
training_set = dataset[:'2016'].iloc[:,1:2].values
test_set = dataset[:'2017'].iloc[:,1:2].values

dataset["High"][:'2016'].plot(figsize=(16,4), legend=True)
dataset["High"]['2017':].plot(figsize=(16,4), legend=True)
plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
plt.title('IBM stock price')
plt.show()

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


#Training data for the LSTM model
x_train = [] #This is a 3D array of shape (2769, 60, 1) where 2769 is the number of training examples, 60 is the number of time steps, and 1 is the number of features (The "high" attribute)
y_train = [] #This is the 1D array of the stock prices that correspond to the last time step in each sequence

for i in range (60, 2769):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



# The LSTM architecture
regressor = Sequential()
# First LSTM Layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM Layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM Layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM Layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN (Recurrent neural network)
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(x_train,y_train,epochs=10, batch_size=32)


#This will get the test set ready in a similar way as the training set
#The following has been done so the first 60 entries of testing set have 60 previous values which is impossible to get unless we take the entire 'High' attribute data for processing
dataset_total = pd.concat((dataset["High"][:'2016'],dataset["High"]['2017':]),axis=0)
#####################inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
start_index = max(0, len(dataset_total) - len(test_set) - 60)  # Prevent negative index
inputs = dataset_total[start_index:].values
print("New shape of inputs:", inputs.shape)
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i,0]) 
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results for the LSTM
plot_predictions(test_set,predicted_stock_price)

# Evaluating our model
return_rmse(test_set,predicted_stock_price)


