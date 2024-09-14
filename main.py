#from model_generate.randomForest import saveData

from data.data_import import data
from config import RFContext, LSTMContext, trainContextLstm

from sklearn.preprocessing import MinMaxScaler

from model_generate.LSTM import LSTMModel

import os
import pandas as pd
import numpy as np

from keras.models import load_model

from menu import menu

if __name__ == '__main__':
    
    option = menu()
    
    if option == '1':
        #os.makedirs(RFContext['savingRoute'], exist_ok=True)
        load = data(**RFContext)
        load.importSave()
    elif option == '2':
        #os.makedirs(LSTMContext['savingRoute'], exist_ok=True)   # Data class' method already does this
        load = data(**LSTMContext)
        load.importSave()
    
    elif option == '3':
        
        lstmModel = LSTMModel(**trainContextLstm)
        
        # Execute the LSTM model and save the model
        rmse = lstmModel.model(route='./model_generate')
        print(rmse)
    elif option == '4':
        
        # get data to predict 
        load = data(**LSTMContext)
        x = load.oneCompanyData(company=trainContextLstm['company'], timeStep=trainContextLstm['timeStep'])[['Close']].values
    
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_x = scaler.fit_transform(x)
        
        # Reshape 
        scaled_x = np.array(scaled_x)
        scaled_x = scaled_x.reshape(1, trainContextLstm['timeStep'],1)
        
        
        # Make tomorrow prediction
        model = load_model('./model_generate/LSTMModel.keras')
        tomorrowPredict = model.predict(scaled_x)
        
        pred = scaler.inverse_transform(tomorrowPredict)[0][0]
        
        
        print(f'Tomorrow the stock price is expected to be: {np.round(pred,2)} US')
    #
    else:
        print('Not a valid input.')
    
