# parts of the code taken from:
# https://nbviewer.ipython.org/github/twiecki/financial-analysis-python-tutorial/blob/master/1.%20Pandas%20Basics.ipynb
# With the consent of the original author, by Thomas Wiecki
import datetime

import pandas as pd
from pandas_datareader import data  # pip install pandas-datareader
from pandas import Series

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('figure', figsize=(8, 7))

labels = ['a', 'b', 'c', 'd', 'e']
s = Series([1, 2, 3, 4, 5], index=labels)
print(s)

print('b' in s)
print(s['b'])

mapping = s.to_dict()
print(mapping)
print(Series(mapping))

aapl = data.get_data_yahoo('AAPL', start=datetime.datetime(2010, 10, 1),
                           end=datetime.datetime(2023, 1, 1))
print(aapl.head())
# See in debug mode. So much more efficient than jupyter notebook.

# dump to a csv:
aapl.to_csv('aapl_ohlc.csv')

# create a DataFrame from that csv
df = pd.read_csv('aapl_ohlc.csv', index_col='Date', parse_dates=True)
print(df.head())
print(df.index)

ts = df['Close'][-10:]  # time series
print(ts)



example1 = df[['Open', 'Close']].head()  # breakpoint here and see slicing in data view

# Adding a new column on the spot can also be done in the console while in debug session.
# Note that writing code in the editor window while in the debug session won't impact the execution flow
# of your current run.
df['diff'] = df.Open - df.Close  # breakpoint here and see in data view
print(df.head())

# some financial quick maths happening below
# Adjusted close is the closing price after adjustments for all applicable splits and dividend distributions
close_px = df['Adj Close']  # timestamp series
mavg = close_px.rolling(window=40).mean()
print(mavg[-10:])

rets = close_px / close_px.shift(1) - 1
print(rets.head())
# whew, quick financial maths done

plt.plot(df['Adj Close'], label="AAPL")
plt.plot(mavg, label="MAVG")
plt.legend()
plt.show()  # debug: color picker for matplotlib colors lovers

pass
# TODO: see stock price chart, experiment with other stock, e.g. INTC

# a fancy chart
df = data.get_data_yahoo(['AAPL', 'GE', 'GOOG', 'IBM', 'KO', 'MSFT', 'PEP'],
                               start=datetime.datetime(2010, 1, 1),
                               end=datetime.datetime(2013, 1, 1))['Adj Close']
print(df.head())
rets = df.pct_change()

plt.scatter(rets.PEP, rets.KO)
plt.xlabel('Returns PEP')
plt.ylabel('Returns KO')
plt.show()

# even more fancy chart
plot2 = pd.plotting.scatter_matrix(rets, diagonal='kde', figsize=(10, 10))
plt.show()
