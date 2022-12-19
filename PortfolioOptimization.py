# all of tyhe libaries which are necessary for the functionality of the program
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from functools import reduce
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# this is just pandas options used for displaying columns which was necessary
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# sets the time for the data which is being used to optimize the users portfolio
# uses the current date for the start and imports data from the past three years on the specified stocks
startTime = datetime.datetime.today() - datetime.timedelta(days = 1095)
endTime = datetime.datetime.today()

# shows the user the start and end time that is being taken into consideration for the evaluation
print(startTime)
print(endTime)


# this pulls the information for each individual stock that is being analyzed
def singleStockPull(ticker):
    yf.pdr_override()
    stockData = pdr.get_data_yahoo(
        [ticker], start=startTime, end=endTime)
    stockData[f'{ticker}'] = stockData["Close"]
    stockData = stockData[[f'{ticker}']]
    return stockData

# this function calls the singleStockPull multiple times to perform the function on each stock
def multipleStockData(tickers):
    df = []
    for i in tickers:
        df.append(singleStockPull(i))
    df = reduce(lambda left, right: pd.merge(
        left, right, on=['Date'], how='outer'), df)
    return df

# calls for the user to provide the tickers which will be used to perform analysis
input_string = input('Give me a list of stocks which you are interested in seperated by blank spaces and I will attempt'
                     'to give you the most optimal arrangement of your investments according to recent stock data ')
stocks_list = input_string.split()
userGivenStocks = stocks_list
print(stocks_list + ' these are the stocks which you chose!')
print("Thank you! Here is an analysis of your stocks! ")

# these stocks were chosen for debugging and testing
# investments = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "SPY", "WMT"]
# gets stock price history for each of the ones named by the user
userPortfolio = multipleStockData(userGivenStocks)

# determines expected return from historical returns
meanHistoricalReturn = mean_historical_return(userPortfolio)
# estimates covariance matrix
ledoitWolfShrinkage = CovarianceShrinkage(userPortfolio).ledoit_wolf()

efficiency = EfficientFrontier(meanHistoricalReturn, ledoitWolfShrinkage)
weights = efficiency.max_sharpe()
cleanedWeights = efficiency.clean_weights()
print(cleanedWeights)

# Based on weights, determine optimal allocation given total portfolio value
efficiency.portfolio_performance(verbose=True)
latestPrices = get_latest_prices(userPortfolio)
alloc = DiscreteAllocation(weights, latestPrices, total_portfolio_value=10000)
alloc, leftover = alloc.lp_portfolio()
print("Discrete allocation:", alloc)
