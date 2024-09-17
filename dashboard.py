import matplotlib.pyplot as plt
import pandas as pd

class dashboard():
    def __init__(self, data) -> None:
        self.data = data
        pass
    
    
    def historicalStock(self):
        plt.style.use('seaborn-v0_8')
        
        fig, ax = plt.subplots()
        
        ax.plot(self.data['Close'], label='Close price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price US')
        ax.set_title('Stock Prices Over Time')
        ax.legend()
        
        return fig
    
    def movingAverages(self):
        plt.style.use('seaborn-v0_8')
        
        
        # Calculate the Moving Average for all companies
        maDay = [10,20,30]

        for ma in maDay:
            name = f'MA {ma} days'
            self.data[name] = self.data['Close'].rolling(ma).mean()
        
        fig, ax = plt.subplots()
        
        for j in maDay:
            ax.plot(self.data[f'MA {j} days'], label=f'MA {j} days')
    
        ax.plot(self.data['Close'], color = 'orange',label='Close price')
        plt.xlabel('Date')
        plt.ylabel('Stock Price US')
        plt.title('Close Moving Averages')
        plt.legend()
        
        return fig
    
    def PredVsActual(self):
        
        pass
    
    def errorMetric(self):
        
        pass
    
    def futurePred(self):
        pass
    