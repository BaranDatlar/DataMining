import requests
import json
import os


class StockMarketData(object):

    _instance = None
    
    def __init__(self,savedJSONFile):
        self.apikey = "1UF6H40ILOX86K5G"
        self.symbol = "IBM" #stock
        self.outputsize = "full" #20 year daily historical data
        self.url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&apikey={self.apikey}&outputsize={self.outputsize}'
        self.savedJSONFile = savedJSONFile

    def __new__(cls,savedJSONFile=None):
        if (cls._instance is None):
            cls._instance = super(StockMarketData, cls).__new__(cls)
            cls._instance.savedJSONFile = savedJSONFile
    
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
        if os.path.exists(self.savedJSONFile):
            with open(self.savedJSONFile, 'r') as file:
                data = json.load(file)
        else:
            response = requests.get(self.url)
            data = response.json()
            with open(self.savedJSONFile, 'w') as file:
                json.dump(data, file)
        return data