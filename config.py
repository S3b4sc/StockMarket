from datetime import datetime
from dateutil.relativedelta import relativedelta


# Random Forest Data Config
RFContext = {
        'savingRoute': 'data/randomForestData/',
        'fileName': 'randomDorestData.h5',
        'companyList': ['AAPL', 'GOOG', 'MSFT', 'AMZN'],
        'startTime': datetime.now() - relativedelta(years=5),
        'endTime': datetime.now()
    }


# LSTM Data Config
LSTMContext = {
        'savingRoute': 'data/lSTMData/',
        'fileName': 'lSTMData.h5',
        'companyList': ['AAPL'],
        'startTime': datetime.now() - relativedelta(years=12),
        'endTime': datetime.now()
    }