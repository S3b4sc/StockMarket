from datetime import datetime
from dateutil.relativedelta import relativedelta


# Random Forest Data Config to generate data
RFContext = {
        'savingRoute': 'data/randomForestData/',
        'fileName': 'randomDorestData.h5',
        'companyList': ['AAPL', 'GOOG', 'MSFT', 'AMZN'],
        'startTime': datetime.now() - relativedelta(years=5),
        'endTime': datetime.now()
    }


# LSTM Data Config to generate data
LSTMContext = {
        'savingRoute': 'data/lSTMData/',
        'fileName': 'lSTMData.h5',
        'companyList': ['AAPL'],
        'startTime': datetime.now() - relativedelta(years=12),
        'endTime': datetime.now()
    }

# LSTM context for training

trainContextLstm = {
    'load_route': './data/lSTMData/lSTMData.h5',                   
    'company': 'AAPL',
    'trainDataPercentage': 0.95,
    'timeStep': 40,
    'epochs': 1,
    'batch_size': 1,
    'lr': 0.01 
}

# LSTM contect for production prediction
predContextLstm = {            
    'company': 'AAPL',
    'days': 20,
    'timeStep': 40
}



