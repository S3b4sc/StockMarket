from datetime import datetime
from dateutil.relativedelta import relativedelta


#Random Forest Data Config
RFContext = {
        'savingRoute': 'data/randomForestData/',
        'fileName': 'randomDorestData.h5',
        'companyList': ['AAPL', 'GOOG', 'MSFT', 'AMZN'],
        'startTime': datetime.now() - relativedelta(years=3),
        'endTime': datetime.now()
    }
