import pandas as pd
import yfinance as yf
from datetime import datetime
#from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay  # Business Day offset
import os

class data:
    def __init__(self, savingRoute:str,fileName:str, companyList:list, startTime:datetime, endTime:datetime) -> None:
        self.savingRoute = savingRoute
        self.name = fileName
        self.companyList = companyList
        self.start = startTime
        self.end = endTime
        
    def importSave(self):
        os.makedirs(self.savingRoute, exist_ok=True)
        
        with pd.HDFStore(self.savingRoute + self.name, mode='w') as store:
            for stock in self.companyList:
                store.put(stock,yf.download(stock, start=self.start, end=self.end), format='table')
    
    def oneCompanyData(self, company:str,timeStep:int):
        final = datetime.now()
        inicio = final - BDay(timeStep+1)
        data = yf.download(company,start=inicio, end=final)
        
        return data
        
        
        