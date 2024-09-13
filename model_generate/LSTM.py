import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam


class LSTM():
    def __init__(self,load_route:str,company:str, trainDataPercentage:float, setimeStep:int, epochs:int, batch_size:int, lr:float) -> None:
        self.load_route = load_route
        self.company = company
        self.trainDataPercentage = trainDataPercentage
        self.setimeStep = setimeStep
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def load_data(self):
        data = pd.read_hdf(self.load_route,self.company)
        return data
        
    def process_sequences(self):
        
        data = self.load_data()
        
        x = []
        y = []

        for i in range(len(data) - self.setimeStep):
            x.append(data[i:(i + self.setimeStep)])
            y.append(data[i + self.setimeStep])
        
        return np.array(x), np.array(y)
        
    
    def load_process_split_data(self):
        # Load the data
        data = pd.read_hdf(self.route, self.company)[['Close']]
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        
        # take training data
        trainDataLen = int(len(data) * self.trainDataPercentage)
        trainData = scaled_data[:trainDataLen, 0]
        x_train, y_train = self.process_sequences(data=trainData,timeStep=self.timeStep)
        
        # Reshape data to train (samples, timeSteps, features)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
        
        return x_train,y_train, scaled_data, scaler
    
    
    def model(self,route:str):
        
        # Create model
        model = Sequential()

        model.add(Input(shape=(self.timeStep, 1)))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(60, return_sequences=False))

        model.add(Dense(22))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=self.lr)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Get the data
        x_train, y_train, scaled_data, scaler = self.load_process_split_data()
        
        # train model
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        
        # Test model
        # Create training dataset
        testData = scaled_data[(self.trainDataLen - self.timeStep):,0]

        # Create testing sequences

        #y_test = data[trainDataLen:]
        x_test,y_test = self.process_sequences(data=testData,timeStep=self.timeStep)


        #                      (samples, timeSteps, features)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_real = scaler.inverse_transform(y_test.reshape(-1,1))

        # Get RMSE
        rmseValue = mean_squared_error(y_true=y_real,y_pred=predictions)
        
        model.save(route + '/LSTMModel.h5')

    
    

