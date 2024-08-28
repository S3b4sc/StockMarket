from data.data_import import data
from datetime import datetime
from dateutil import relativedelta
import os


def saveData():
    context = {
        'savingRoute': 'randomForest',
        'fileName': 'randomDorestData.h5',
        'companyList': ['AAPL', 'GOOG', 'MSFT', 'AMZN'],
        'startTime': datetime.now() - relativedelta(years=3),
        'endTime': datetime.now()
    }
    
    os.makedirs(context['savingRoute'], exist_ok=True)
    
    save = data(**context)
    save.importSave()







