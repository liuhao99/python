# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:37:04 2020

@author: a0866568
"""

import yfinance as yf
import numpy as np
import requests

import pandas as pd
## from get_all_tickers import get_tickers as gt


# need pip install statsmodels==0.9
import statsmodels.api as sm

from statistics import mean

from pandas import DataFrame
import matplotlib.pyplot as plt

import matplotlib.dates as dates
#plt_dates = dates.date2num(t.to_pydatetime())



ticker=["MSFT", "AAPL", "IBM", "GOOG"]

## get historical market data
#msft = yf.Ticker("MSFT")
#hist = msft.history(period="max")
hist = yf.download(ticker)

## dtype='datetime64[ns]', name='Date', => .loc(), .iloc(), resample('BM') work
hist.index 

hist.index = pd.to_datetime(hist.index)
mask = (hist.index > '2006-10-01') & (hist.index <= '2012-01-01')
data = hist.loc[mask]

daily_close_px = data['Adj Close']
daily_pct_change = daily_close_px.pct_change()
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

pd.plotting.scatter_matrix(daily_pct_change, alpha=0.1,figsize=(12,12))

aapl = daily_close_px[['AAPL']]
msft = daily_close_px[['MSFT']]
aapl_daily = daily_close_px['AAPL']
aapl['42'] = aapl_daily.rolling(window=40).mean()
aapl['252'] = aapl_daily.rolling(window=252).mean()

aapl.index = pd.to_datetime(aapl.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
aapl.plot()

min_periods = 75
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
vol.plot(figsize=(10,8)) 


all_returns = np.log(daily_close_px / daily_close_px.shift(1))
aapl_returns = all_returns['AAPL']
msft_returns = all_returns['MSFT']
##add [1:] to the concatenation of the AAPL and MSFT return data so that you don’t have any NaN values that can interfere with your model.
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']
return_data.info()
X = sm.add_constant(return_data['AAPL'])
model = sm.OLS(return_data['MSFT'],X).fit()
print(model.summary())


# Plot returns of AAPL and MSFT
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')

# Add an axis to the plot
ax = plt.axis()

# Initialize `x`
x = np.linspace(ax[0], ax[1] + 0.01)

# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)

# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
plt.show()

# rolling correlation
return_data.index = pd.to_datetime(return_data.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()



short_window = 40
long_window = 100

signals = pd.DataFrame(index = aapl.index)
signals['signal']=0.0

signals['short_mavg'] = aapl['AAPL'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = aapl['AAPL'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

signals['positions']=signals['signal'].diff()

# signal.index from object to datetime64[ns]
signals.index = pd.to_datetime(signals.index)
# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl['AAPL'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

plt.show()


aapl['Adj Close']=data['Adj Close'][['AAPL']]
# Set the initial capital
initial_capital= float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions['AAPL'] = 100*signals['signal']   
  
# Initialize the portfolio with value owned   
portfolio = positions.multiply(aapl['Adj Close'], axis=0)

# Store the difference in shares owned 
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()   

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
print(portfolio.head())


# https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/multiple_figs_demo.html
fig, axs = plt.subplots(2)
axs[0].plot(portfolio['cash'])
axs[1].plot(portfolio['holdings'])


# Isolate the returns of your strategy
returns = portfolio['returns']

# annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Print the Sharpe ratio
print(sharpe_ratio)


# Define a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day 
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl['Adj Close']/rolling_max - 1.0

# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()

# Show the plot
plt.show()

# Get the number of days in `aapl`
days = (aapl.index[-1] - aapl.index[0]).days

# Calculate the CAGR 
cagr = ((((aapl['Adj Close'][-1]) / aapl['Adj Close'][1])) ** (365.0/days)) - 1

# Print the CAGR
print(cagr)


#  fig, axs = plt.subplots(2)

# Calculate the sum
#
## show actions (dividends, splits)
#msft.actions
#
## show dividends
#msft.dividends
#
## show splits
#msft.splits
#
## show financials
#msft.financials
#msft.quarterly_financials
#
## show major holders
#msft.major_holders
#
## show institutional holders
#msft.institutional_holders
#
## show balance sheet
#msft.balance_sheet
#msft.quarterly_balance_sheet
#
## show cashflow
#msft.cashflow
#msft.quarterly_cashflow
#
## show earnings
#msft.earnings
#msft.quarterly_earnings
#
## show sustainability
#msft.sustainability
#
## show analysts recommendations
#msft.recommendations
#
## show next event (earnings, etc)
#msft.calendar
#
## show ISIN code - *experimental*
## ISIN = International Securities Identification Number
#msft.isin
#
## show options expirations
#msft.options
#
## get option chain for specific expiration
#opt = msft.option_chain('YYYY-MM-DD')
## data available via: opt.calls, opt.puts

# Calculating the short-window simple moving average

# dtype='datetime64[ns]' to dtype='object'
# dtype='object', name='Date' => plot() works
hist.index = pd.to_datetime(hist.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
hist.index

# dtype='object' to dtype='datetime64[ns]'
hist.index = pd.to_datetime(hist.index, infer_datetime_format=True)



hist.head()
hist.describe()

msft = hist['Close']

#print(msft.loc["2008"].head())
#
#type(msft)
# Daily returns
#daily_pct_change = daily_close / daily_close.shift(1) - 1
#or pct_change()
daily_close=hist[['Adj Close']]
daily_pct_change = daily_close.pct_change()
daily_pct_change.fillna(0,inplace=True)

monthly = hist.resample('BM').apply(lambda x:x[-1])

cum_daily_return = (1+ daily_pct_change).cumprod()
#msft_daily_returns = (msft / msft.shift(1)) - 1
#msft.plot(grid=True)
#msft_daily_returns.plot()
#msft_daily_returns.plot.hist(bins = 60)
#plt.xlabel("Date")
#plt.ylabel("Adjusted")
#plt.title("Miscrosoft Price data")
#plt.style.use('dark_background')
#plt.show()

#
#short_rolling_msft = msft.rolling(window=20).mean()
#short_rolling_msft.head(20)
#long_rolling_msft = msft.rolling(window=100).mean()
#
#ema_short_msft = msft.ewm(span=20, adjust=False).mean()




