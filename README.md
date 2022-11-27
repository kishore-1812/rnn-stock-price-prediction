# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
The stock market is generally very unpredictable in nature.The overall challenge is to determine the gradient difference between one Opening price and the next. Recurrent Neural Network (RNN) algorithm is used on time-series data of the stocks. The predicted closing prices are cross checked with the true closing price.

<img width="598" alt="image" src="https://user-images.githubusercontent.com/63336975/196042594-315507da-a687-4db7-9da2-97aedff0ee2a.png">


## Neural Network Model

<img width="373" alt="image" src="https://user-images.githubusercontent.com/63336975/196042692-3fcf14ab-40fb-4b89-bc38-f423e22fae5a.png">

## DESIGN STEPS

### STEP 1:
Download and load the dataset to colab.

### STEP 2:
Scale the data using MinMaxScaler

### STEP 3:
Split the data into train and test.

### STEP 4:
Build the recurrent neural network

### STEP 5:
Train the model with training data

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Plot the Stock prediction plot

## PROGRAM
``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = Sequential([layers.SimpleRNN(70,input_shape=(60,1)),layers.Dense(1)])
model.compile(optimizer='Adam', loss='mae')
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

<img width="299" alt="image" src="https://user-images.githubusercontent.com/63336975/196042357-28bef110-2641-4e6b-9530-a5dc11fb7865.png">

### Mean Square Error

<img width="410" alt="image" src="https://user-images.githubusercontent.com/63336975/196042407-8c8e596f-4adf-4764-81a6-d512ae28a5c6.png">

## RESULT
Successfully developed a Recurrent neural network for Stock Price Prediction.
