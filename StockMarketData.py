import requests

class StockMarketData(object):

    _instance = None
    
    def __init__(self):
        self.apikey = "PUSAP5EGEX3G4OJ1"
        self.symbol = "IBM" #stock
        self.outputsize = "full" #20 year daily historical data
        self.url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&apikey={self.apikey}&outputsize={self.outputsize}'

    def __new__(cls):
        if (cls._instance is None):
            cls._instance = super(StockMarketData, cls).__new__(cls)
    
        return cls._instance

    
    def ExtractDataCSV(self):
        r = requests.get(self.url)
        if r.status_code == 200:
            csvData = r.text
            with open('HistoricalData.csv', 'w') as file:
                file.write(csvData)
        else:
            print("Transaction failed:", r.status_code)   

    def ExtractDataJSON(self):
        r = requests.get(self.url)
        data = r.json()
        return data