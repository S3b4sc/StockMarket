import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam


class LSTMModel():
    """
    A class to manage the training and testing of an LSTM model for stock price prediction.

    Attributes:
        load_route (str)
        company (str) 
        trainDataPercentagata to use for training.
        timeStep (int)
        epochs (int)
        batch_size (int)
        lr (float)
    """
    
    
    def __init__(self,load_route:str,company:str, trainDataPercentage:float, timeStep:int, epochs:int, batch_size:int, lr:float) -> None:
        
        """
        Initializes the LSTMModel class with configuration for model training.

        Args:
            load_route (str): Path to the HDF5 file containing stock data.
            company (str): The company name (or identifier) to load data for.
            trainDataPercentage (float): The percentage of data used for training.
            timeStep (int): Number of time steps used in the LSTM input.
            epochs (int): Number of epochs for training.
            batch_size (int): The batch size for training.
            lr (float): Learning rate for the optimizer.
        """
        
        self.load_route = load_route
        self.company = company
        self.trainDataPercentage = trainDataPercentage
        self.timeStep = timeStep
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def load_data(self):
        
        """
        Loads the stock closing prices from the specified HDF5 file.

        Returns:
            np.ndarray: A numpy array containing the closing prices for the specified company.
        """
        
        data = pd.read_hdf(self.load_route,self.company)['Close'].values
        return data
        
    def process_sequences(self,data:np.ndarray):
        
        """
        Processes the stock data into sequences of features and corresponding labels for LSTM.

        Args:
            data (np.ndarray): A numpy array of stock prices.

        Returns:
            tuple: Two numpy arrays containing input sequences (X) and target values (y).
        """
        
        x = []
        y = []

        for i in range(len(data) - self.timeStep):
            x.append(data[i:(i + self.timeStep)])
            y.append(data[i + self.timeStep])
        
        return np.array(x), np.array(y)
        
    
    def load_process_split_data(self):
        
        """
        Loads, scales, and splits the data into training sequences for the LSTM model.

        Returns:
            tuple: 
                - x_train (np.ndarray): Training input sequences.
                - y_train (np.ndarray): Training target values.
                - scaled_data (np.ndarray): Scaled stock price data.
                - scaler (MinMaxScaler): Scaler used for normalizing the data.
                - trainDataLen (int): Length of the training data.
        """
        
        # Load the data
        data = pd.read_hdf(self.load_route, self.company)[['Close']].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        
        # take training data
        trainDataLen = int(len(data) * self.trainDataPercentage)
        trainData = scaled_data[:trainDataLen, 0]
        x_train, y_train = self.process_sequences(data=trainData)
        
        # Reshape data to train (samples, timeSteps, features)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
        
        return x_train,y_train, scaled_data, scaler, trainDataLen
    
    
    def model(self,route:str):
        
        """
        Builds, trains, and tests the LSTM model, and saves it to the specified route.

        Args:
            route (str): The directory to save the trained LSTM model.

        Returns:
            float: The Root Mean Squared Error (RMSE) of the model's predictions on the test set.
        """
        
        
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
        x_train, y_train, scaled_data, scaler, trainDataLen = self.load_process_split_data()
        
        # train model
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        
        # Test model
        # Create training dataset
        testData = scaled_data[(trainDataLen - self.timeStep):,0]

        # Create testing sequences

        #y_test = data[trainData:]
        x_test,y_test = self.process_sequences(data=testData)


        #                      (samples, timeSteps, features)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_real = scaler.inverse_transform(y_test.reshape(-1,1))

        # Get RMSE
        rmseValue = np.sqrt(mean_squared_error(y_true=y_real,y_pred=predictions))
        
        model.save(route + '/LSTMModel.keras')
        
        return rmseValue

    
    

