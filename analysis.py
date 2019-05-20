
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import pandas as pd
import pandas_datareader.data as web

#Loading data
company='AAPL'

plt.rcParams['figure.figsize'] = [20, 10]
start= dt.datetime(2000,1,1)
end=dt.datetime(2019,12,31)

df = web.DataReader(company,'yahoo',start,end)
df.to_csv(company+'.csv')

pd.read_csv(company+'.csv',parse_dates=True,index_col=0)
df.head()


style.use('ggplot')
df['Adj Close'].plot()
plt.show()


df_copy=df.copy()
df_copy['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()

df_ohlc=df['Adj Close'].resample('10D').ohlc()
df_volume=df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)


ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date()
ax1.plot(df_copy.index,df_copy['100ma'])
candlestick_ohlc(ax1, df_ohlc.values,width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0,facecolor='green', alpha=0.5)
plt.show()