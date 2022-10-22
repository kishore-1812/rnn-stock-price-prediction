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
Build the convolutional neural network

### STEP 5:
Train the model with training data

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Plot the Stock prediction plot

## PROGRAM
https://github.com/kishore-1812/rnn-stock-price-prediction/blob/main/Ex05_StockPricePrediction_063.ipynb

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

<img width="299" alt="image" src="https://user-images.githubusercontent.com/63336975/196042357-28bef110-2641-4e6b-9530-a5dc11fb7865.png">

### Mean Square Error

<img width="410" alt="image" src="https://user-images.githubusercontent.com/63336975/196042407-8c8e596f-4adf-4764-81a6-d512ae28a5c6.png">

## RESULT
Successfully developed a Recurrent neural network for Stock Price Prediction.
