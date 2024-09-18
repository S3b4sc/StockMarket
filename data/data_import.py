import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay  # Business Day offset
import os

class data:
    
    """
    A class to manage the download, processing, and saving of stock market data.

    Attributes:
        savingRoute (str)
        name (str)
        companyList (list)
        start (datetime)
        end (datetime)
    """
    
    def __init__(self, savingRoute:str,fileName:str, companyList:list, startTime:datetime, endTime:datetime) -> None:
        
        """
        Initializes the data class with information about where to save data and what companies to download.

        Args:
            savingRoute (str): Path to the directory where data will be saved.
            fileName (str): The name of the file where the data will be stored.
            companyList (list): A list of company symbols whose stock data will be downloaded.
            startTime (datetime): The start date for the data download.
            endTime (datetime): The end date for the data download.
        """
        
        self.savingRoute = savingRoute
        self.name = fileName
        self.companyList = companyList
        self.start = startTime
        self.end = endTime
        
    def importSave(self):
        
        """
        Downloads stock data for all companies in the companyList and saves it to an HDF5 file.

        Downloads data from Yahoo Finance for the specified date range and saves it for each company.

        Saves the data to a file at the specified saving route.

        Raises:
            OSError: If there is an issue creating the directory for saving the data.
        """
        
        os.makedirs(self.savingRoute, exist_ok=True)
        
        with pd.HDFStore(self.savingRoute + self.name, mode='w') as store:
            for stock in self.companyList:
                store.put(stock,yf.download(stock, start=self.start, end=self.end), format='table')
    
    def oneCompanyData(self, company:str,timeStep:int):
        
        """
        Downloads historical stock data for a single company for the last 'timeStep' business days.

        Args:
            company (str): The stock symbol of the company.
            timeStep (int): The number of business days of data to retrieve.

        Returns:
            DataFrame: A pandas DataFrame containing the company's stock data for the given period.
        """
        
        final = datetime.now()
        inicio = final - BDay(timeStep + 1)
        data = yf.download(company,start=inicio, end=final)
        
        return data
        
    def getDashData(self, company:str):
        
        """
        Downloads historical stock data for a company over the last 12 years.

        Args:
            company (str): The stock symbol of the company.

        Returns:
            DataFrame: A pandas DataFrame containing the company's stock data over the past 12 years.
        """

        final = datetime.now()
        inicio = final - relativedelta(years=12)
        data = yf.download(company,start=inicio, end=final)

        return data
        
        
def predData(company:str, days:int, timeStep:int):
    
        """
        Downloads historical stock data for the past 'days' + 'timeStep' business days.
    
        Args:
            company (str): The stock symbol of the company.
            days (int): The number of days of data to retrieve.
            timeStep (int): The number of time steps to include.
    
        Returns:
            DataFrame: A pandas DataFrame containing the company's stock data for the given period.
        """
    
        final = datetime.now()
        inicio = final - BDay(days + timeStep)       
        
        data = yf.download(company,start=inicio, end=final)
        return data
        
    