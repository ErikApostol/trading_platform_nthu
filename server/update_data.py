import pandas as pd
all_data = pd.read_csv('data_for_trading_platform.csv')
#all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
import yfinance as yf
#from datetime import datetime

tickers = list(all_data.Ticker.unique())
for ticker in tickers:
    current_max_date = all_data[all_data['Ticker']==ticker].Date.max()
    if current_max_date < "2020-10-01":
        new_data = yf.download(ticker, start=current_max_date, end="2020-10-01")
        new_data['Ticker'] = ticker
        new_data['Date'] = new_data.index
        all_data = all_data.append(new_data[new_data['Date']>current_max_date], ignore_index=True)

all_data.to_csv('data_for_trading_platform.csv', index=False, columns=['Date','Ticker','Open','High','Low','Close','Adj Close','Volume'])
