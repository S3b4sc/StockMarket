#from model_generate.randomForest import saveData

from data.data_import import data
from config import RFContext, LSTMContext, trainContextLstm

from model_generate.LSTM import LSTM

import os

from keras.models import load_model

from menu import menu

if __name__ == '__main__':
    
    option = menu()
    
    if option == '1':
        os.makedirs(RFContext['savingRoute'], exist_ok=True)
        load = data(**RFContext)
        load.importSave()
    elif option == 2:
        os.makedirs(LSTMContext['savingRoute'], exist_ok=True)
        load = data(**LSTMContext)
        load.importSave()
    
    elif option == 3:
        
        lstmModel = LSTM(**trainContextLstm)
        
        # Execute the LSTM model
        lstmModel.model(route='./model_generate')
        
    elif option == 4:
        model = load_model('./model_generate/LSTMModel.h5')
    
