from __main__ import *
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from cvxopt import matrix, solvers
from cvxopt.blas import dot
from cvxopt.solvers import qp

# Turn off progress printing
solvers.options['show_progress'] = False

port = tickers

start_dates = [datetime.datetime(2015, 1, 1),
               datetime.datetime(2015, 4, 1),
               datetime.datetime(2015, 7, 1),
               datetime.datetime(2015, 10, 1),
               datetime.datetime(2016, 1, 1),
               datetime.datetime(2016, 4, 1),
               datetime.datetime(2016, 7, 1),
               datetime.datetime(2016, 10, 1),
               datetime.datetime(2017, 1, 1),
               datetime.datetime(2017, 4, 1),
               datetime.datetime(2017, 7, 1),
               datetime.datetime(2017, 10, 1),
               datetime.datetime(2018, 1, 1),
               datetime.datetime(2018, 4, 1),
               datetime.datetime(2018, 7, 1),
               datetime.datetime(2018, 10, 1),
               datetime.datetime(2019, 1, 1),
               datetime.datetime(2019, 4, 1),
               datetime.datetime(2019, 7, 1),
               datetime.datetime(2019, 10, 1),
               datetime.datetime(2020, 1, 1),
               datetime.datetime(2020, 4, 1),
               datetime.datetime(2020, 7, 1) ]

all_data = pd.read_csv('data_for_trading_platform_202007.csv')
all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')

def stockpri(ticker, start, end):
    data = all_data[ (all_data['Ticker']==ticker) & (all_data['Date']>=start) & (all_data['Date']<=end) ]
    data.set_index('Date', inplace=True)
    data = data['Adj Close']
    return data

def result(weight):
    sigma = np.sqrt(np.dot(weight, np.dot(log_returns.cov()*252, weight.T)))
    profit = np.dot(weight, np.exp(log_returns.mean()*252) - 1) + 1
    return np.array([sigma, profit, profit/sigma])

def sigma(weight):
    return result(weight)[0]

portfolio_value = pd.Series([100])

for i in range(len(start_dates)-3):

    ### Take 6 months to backtest ###

    start = start_dates[i]
    end   = start_dates[i+2]

    data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in port })
    data = data.dropna()
    
    returns = data.pct_change() + 1
    returns = returns.dropna()
    log_returns = np.log(data.pct_change() + 1)
    log_returns = log_returns.dropna()
    
    # Markowitz frontier
    profit = np.linspace(0., 3., 100)
    frontier = []
    w = []
    for p in profit:
        # Problem data.
        n = len(port)
        S = matrix(log_returns.cov().values*252)
        pbar = matrix(0.0, (n,1))
        G = matrix(0.0, (2*n,n))
        G[::(2*n+1)] = 1.0
        G[n::(2*n+1)] = -1.0
        h = matrix(1.0, (2*n,1))
        A = matrix(np.concatenate((np.ones((1,n)), np.exp(log_returns.mean()*252).values.reshape((1,n))), axis=0))
        b = matrix([1, p], (2, 1))
        
        # Compute trade-off.
        res = qp(S, -pbar, G, h, A, b)
    
        if res['status'] == 'optimal':
            res_weight = res['x']
            s = math.sqrt(dot(res_weight, S*res_weight))
            frontier.append(np.array([p, s]))
            w.append(res_weight)
    frontier = np.array(frontier)
    x = np.array(frontier[:, 0])
    y = np.array(frontier[:, 1])

    frontier_sharpe_ratios = np.divide(x-1, y)
    optimal_portfolio_index = np.argmax(frontier_sharpe_ratios)
    optimal_weights = w[optimal_portfolio_index]
    

    ### paper trade on the next three months ###

    start = start_dates[i+2]
    end   = start_dates[i+3]

    data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in port })
    data = data.dropna()
    
    returns = data.pct_change() + 1
    returns = returns.dropna()
    #print(returns.keys)
    log_returns = np.log(data.pct_change() + 1)
    log_returns = log_returns.dropna()

    portfolio_value_new_window = portfolio_value.iloc[-1].item() * pd.Series(np.dot(returns, optimal_weights).cumprod())
    portfolio_value_new_window.index = returns.index
    portfolio_value = portfolio_value.append(portfolio_value_new_window) 
    

avg_annual_return = np.exp(np.log(portfolio_value.pct_change() + 1).mean() * 252) - 1
annual_volatility = portfolio_value.pct_change().std() * math.sqrt(252)
sharpe_ratio = avg_annual_return/annual_volatility
max_drawdown = - np.amin(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value)) - 1)

print('The last optimal weights are\n', optimal_weights)
print('Sharpe ratio: ', sharpe_ratio, ', Return: ', avg_annual_return, ', Volatility: ', annual_volatility, ', Maximum Drawdown: ', max_drawdown)

print(portfolio_value)
plt.plot(portfolio_value.iloc[1:])
plt.savefig('static/img/portfolio_values/'+str(strategy_id)+'.png')
