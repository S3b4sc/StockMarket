#from model_generate.randomForest import saveData

from data.data_import import data, predData
from config import RFContext, LSTMContext, trainContextLstm, predContextLstm

from sklearn.preprocessing import MinMaxScaler

from model_generate.LSTM import LSTMModel

import os
import pandas as pd
import numpy as np



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
        st.write('#### Section 1: Close graphs')
        
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
            with st.expander("Setion 1: Moving Averages"):
                st.pyplot(figure2)
    
    with st.container():
        st.write('### Section 2: LSTM Model Testing Results')
        

        # Import data for predictions
        info2 = predData(**predContextLstm)
        
        # Tomorrow's prediction
        
        load = data(**LSTMContext)
        x = load.oneCompanyData(company=trainContextLstm['company'], timeStep=trainContextLstm['timeStep'])
        tomorrowPred = dashboard(data=x)
        
        
        value = tomorrowPred.tomorrowPred(timeStep=predContextLstm['timeStep'] )
        st.write(f'Tomorrow the stock price is expected to be: {value} US')

        predData = dashboard(data=info2)
        figure3,dataframe,rmseValue = predData.futurePred(company=predContextLstm['company'],timeStep=predContextLstm['timeStep'] )

        
        st.pyplot(figure3)
        st.markdown(f"<h2 style='text-align: center;'>Current rmse Value: {rmseValue}</h2>", unsafe_allow_html=True)
        #st.write('Current rmse Value: ', rmseValue)
        with st.expander("Setion 2: Model Training Hyperparmas"):
            st.dataframe(dataframe)
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