import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from keras.models import load_model
import mlflow

class dashboard():
    def __init__(self, data) -> None:
        self.data = data
        pass
    
    
    def historicalStock(self):
        plt.style.use('seaborn-v0_8')
        
        fig, ax = plt.subplots()
        
        ax.plot(self.data['Close'], label='Close price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price US')
        ax.set_title('Stock Prices Over Time')
        ax.legend()
        
        return fig
    
    def movingAverages(self):
        plt.style.use('seaborn-v0_8')
        
        
        # Calculate the Moving Average for all companies
        maDay = [10,20,30]

        for ma in maDay:
            name = f'MA {ma} days'
            self.data[name] = self.data['Close'].rolling(ma).mean()
        
        fig, ax = plt.subplots()
        
        for j in maDay:
            ax.plot(self.data[f'MA {j} days'], label=f'MA {j} days')
    
        ax.plot(self.data['Close'], color = 'orange',label='Close price')
        plt.xlabel('Date')
        plt.ylabel('Stock Price US')
        plt.title('Close Moving Averages')
        plt.legend()
        
        return fig
    
    def tomorrowPred(self,timeStep:int):
        # get data to predict 
        x = self.data[['Close']].values
    
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_x = scaler.fit_transform(x)
        
        # Reshape 
        scaled_x = np.array(scaled_x)
        scaled_x = scaled_x.reshape(1,timeStep,1)
        
        
        # Make tomorrow prediction
        model = load_model('./model_generate/LSTMModel.keras')
        tomorrowPredict = model.predict(scaled_x)
        
        pred = scaler.inverse_transform(tomorrowPredict)[0][0]
        
        return np.round(pred,2)
        
       #st.write(f'Tomorrow the stock price is expected to be: {np.round(pred,2)} US')
    
    def futurePred(self, company:str,timeStep:int):
        
        def process_sequences(data:np.ndarray):

            x = []
            y = []

            for i in range(len(data) - timeStep):
                x.append(data[i:(i + timeStep)])
                y.append(data[i + timeStep])

            return np.array(x), np.array(y)
        
        # Get data to predict 
        x = self.data[['Close']].values
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_x = scaler.fit_transform(x)
        
        x_test,y_test = process_sequences(data=scaled_x)

        # Reshape 
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


        # Make tomorrow prediction
        model = load_model('./model_generate/LSTMModel.keras')
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_real = scaler.inverse_transform(y_test.reshape(-1,1))

        
        predicitions = model.predict(x_test)
        pred = scaler.inverse_transform(predicitions).reshape(-1)
        
        # Get RMSE
        rmseValue = np.sqrt(mean_squared_error(y_true=y_real, y_pred=predictions))
        
        # Get experiment with the prod model info
        experiment = mlflow.get_experiment_by_name("LSTM_experiment_1")
        # Get the runs
        runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
        prodRun = runs.iloc[0][['metrics.rmse','status','params.epohcs','params.Learning rate','params.Train percentage']]
        
        # Metrics DataFrame
        #dataDict = {
        #    'RMSE': rmseValue,
        #    ''
        #}
        

        fig, ax = plt.subplots()
        
        ax.plot(self.data.index[-18:],pred, label='Predicted Close price')
        ax.plot(self.data['Close'],label='Close price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price US')
        ax.set_title('Stock Prices Over Time')
        ax.legend()
        
        return fig,prodRun, rmseValue
    