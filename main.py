#from model_generate.randomForest import saveData

from data.data_import import data
from config import RFContext, LSTMContext

import os

from menu import menu

if __name__ == '__main__':
    
    option = menu()
    
    if option == '1':
        os.makedirs(RFContext['savingRoute'], exist_ok=True)
        load = data(**RFContext)
        load.importSave()
    else:
        os.makedirs(LSTMContext['savingRoute'], exist_ok=True)
        load = data(**LSTMContext)
        load.importSave()
    
