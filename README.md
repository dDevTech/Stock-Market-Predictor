# Stock-Market-Predictor
Stock Market Predictor with LSTM network. Web scraping and analyzing tools (ohlc, mean)

With this stock market predictor you will be able to analyze almost 500 companies and their future. This project includes a WebScraping tool to get the stock data from different companies.

Also you will be able to plot different finance graphs of the OHLC or the mean of the AdjClose

# Requirements
The program uses Python 3.7.3

There are python files and jupyter notebooks

Modules used: numpy, pandas, mpl_finance, matplotlib, keras, pickle...

# Scraping the data
The file of web scraping is **StockScraper.py**

-Execute the function **save_sp500_tickers()** to get the 500 companies tickers and save on a file format .pickle

-Execute **get_data()** to use the yahoo_finance API and get the data with Pandas from the different companies symbols (required first step) and save it on a folder called stock_dfs

-Finally execute **compile_data()** to create the  csv of all the companies data for the correlation

This steps are esential for the next steps

Optional

-Use **visualize_data()** to see the plot of the correlation between companies (it will take some time)

# Training the netwok
The file of neural networkm predictor is is **AIFinance.py**

-To train and test a company you should use the function **predictFuture('AAPL')**
You can use whatever company symbol you want

#Analyze data
Use the jupyter notebook **StockMarket.ipynb**

You will be able to see step by step the different analyze tools and also the training of a company

The company can be change at the bottom of the notebook
