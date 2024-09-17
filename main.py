#from model_generate.randomForest import saveData

from data.data_import import data
from config import RFContext, LSTMContext, trainContextLstm

from sklearn.preprocessing import MinMaxScaler

from model_generate.LSTM import LSTMModel

import os
import pandas as pd
import numpy as np

from keras.models import load_model

import streamlit as st
#--------------------------------------------------
from dashboard import dashboard
#--------------------------------------------------


if __name__ == '__main__':
    
    # Get the data
     # Import Historical Close data
    load = data(**LSTMContext)
    info = load.getDashData(company='AAPL')
        
    st.title('Stock Price Prediction Dashboard')
    
    # Set Dashboard distribution
    
    with st.container():
        st.write('### Section 1: Close graphs')
        
        # Let the user select the time frame
        min_date = LSTMContext['startTime']
        max_date = LSTMContext['endTime']

        date_range = st.slider("Select Date Range", min_date, max_date, (min_date, max_date))
        
        filteredData = info.loc[date_range[0]:date_range[1]]
        
        col1, col2 = st.columns(2)
        
        # Ceate dhasboar object with the filtered data
        historical = dashboard(data=filteredData)
        figure1 = historical.historicalStock()
        figure2 = historical.movingAverages()
            
        with col1:
            st.pyplot(figure1)

        with col2:
            st.pyplot(figure2)
    
    # With st.container():
        st.write('### SEction 1: Close claphs')
        
        col3, col4 = st.columns(2)

        # Import Historical Close data
        load = data(**LSTMContext)
        info = load.getDashData(company='AAPL')

        historical = dashboard(data=info)
        figure1 = historical.historicalStock()

        with col1:
            st.pyplot(figure1)
    #menu()
    #
    #if st.button('1'):
    #    #os.makedirs(RFContext['savingRoute'], exist_ok=True)
    #    load = data(**RFContext)
    #    load.importSave()
    #elif st.button('2'):
    #    #os.makedirs(LSTMContext['savingRoute'], exist_ok=True)   # Data class' method already does this
    #    load = data(**LSTMContext)
    #    load.importSave()
    #
    #elif st.button('3'):
    #    
    #    lstmModel = LSTMModel(**trainContextLstm)
    #    
    #    # Execute the LSTM model and save the model
    #    rmse = lstmModel.model(route='./model_generate')
    #    print(rmse)
    #elif st.button('4'):
    #    
    #    # get data to predict 
    #    load = data(**LSTMContext)
    #    x = load.oneCompanyData(company=trainContextLstm['company'], timeStep=trainContextLstm['timeStep'])[['Close']].values
    #
    #    # Normalize the data
    #    scaler = MinMaxScaler(feature_range=(0,1))
    #    scaled_x = scaler.fit_transform(x)
    #    
    #    # Reshape 
    #    scaled_x = np.array(scaled_x)
    #    scaled_x = scaled_x.reshape(1, trainContextLstm['timeStep'],1)
    #    
    #    
    #    # Make tomorrow prediction
    #    model = load_model('./model_generate/LSTMModel.keras')
    #    tomorrowPredict = model.predict(scaled_x)
    #    
    #    pred = scaler.inverse_transform(tomorrowPredict)[0][0]
    #    
    #    
    #    st.write(f'Tomorrow the stock price is expected to be: {np.round(pred,2)} US')
#
    #
#