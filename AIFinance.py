import numpy as np
import pandas as pd
import  pickle
from sklearn import svm,model_selection,neighbors
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from collections import Counter
import os
import matplotlib.pyplot as plt


def process_data_for_labels(ticker):
    hm = 7
    df = pd.read_csv('sp500_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)

    for i in range(1, hm+1):
        df[f'{ticker}_{i}d']=(df[ticker].shift(-i)-df[ticker])/df[ticker]

    df.fillna(0,inplace=True)
    return tickers,df
def buy_sell_hold(*args):
    cols= [c for c in args]
    requirement =0.25
    for col in cols:
        if(col>requirement):
            return 1
        if(col<-requirement):
            return -1
    return 0
def extract_feature_sets(ticker):
    tickers,df = process_data_for_labels(ticker)

    df[f'{ticker}_target']=list(map(buy_sell_hold,
                                    df[f'{ticker}_1d'],
                                    df[f'{ticker}_2d']
                                    ,df[f'{ticker}_3d']
                                    ,df[f'{ticker}_4d']
                                    ,df[f'{ticker}_5d']
                                    ,df[f'{ticker}_6d']
                                    ,df[f'{ticker}_7d']))

    vals= df[f'{ticker}_target'].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread', Counter(str_vals))

    df.fillna(0,inplace=True)

    df= df.replace([np.inf,-np.inf],np.nan)
    df.dropna(inplace=True)

    df_vals= df[[ticker for ticker in tickers]].pct_change()
    df_vals=df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)

    X = df_vals.values
    y = df[f'{ticker}_target'].values

    return X,y,df

def do_ml(ticker):
    X,y,df = extract_feature_sets(ticker)
    X_train,X_test, y_train,y_test=model_selection.train_test_split(X,y,test_size=0.25)

   # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])
    clf.fit(X_train,y_train)
    confidence = clf.score(X_test,y_test)
    print('Accuracy',confidence)
    prediction = clf.predict(X_test)
    print('Predicted spread: ', Counter(prediction))
    return confidence
def create_dataset(ticker, daysPrediction=50,splitProportion=0.8):
    #Read data
    df = pd.read_csv('sp500_closes.csv')
    tickers = df.columns.values.tolist()


    #Get Adj Close values of ticker and convert it to numpy array
    values = df[ticker].values
    values=values.reshape(-1,1)

    #Split the dataset in the training set and test set
    dataset_train = np.array(values[:int(values.shape[0] * splitProportion)])
    dataset_test = np.array(values[int(values.shape[0] * splitProportion) - daysPrediction:])

    #Normalize dataset in range [0-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    #Create the X and Y variables of the dataset in a format numpy array
    x_train, y_train = create_d_dataset(dataset_train,daysPrediction)
    x_test, y_test = create_d_dataset(dataset_test,daysPrediction)

    #Prepare data for the neural network
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Return dataset and other data
    return x_test,y_test,x_train,y_train,df,ticker,scaler,values
def create_d_dataset(val,daysPrediction):

    x = []
    y = []
    for i in range(daysPrediction, val.shape[0]):
        x.append(val[i-daysPrediction:i, 0])
        y.append(val[i, 0])
    x = np.array(x)
    y = np.array(y)

    return x,y


def train(train_x,train_y,ticker):
    #Create LSTM RNN model
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    #Train network and save it
    if (not os.path.exists(f'stock_prediction-{ticker}.h5')):
        model.fit(train_x, train_y, epochs=3, batch_size=32)
        model.save(f'stock_prediction-{ticker}.h5')

def show_results(test_x,test_y,train_y,ticker,scaler,values,df):
    #Load the trained neural network
    model = load_model(f'stock_prediction-{ticker}.h5')

    #Use the network to to predict the test dataset
    predictions = model.predict(test_x)
    print(model.evaluate(test_x,test_y))
    print(test_x)
    predictions = scaler.inverse_transform(predictions)
    print(values)

    #Graphically see the prediction of the neural network
    fig, ax = plt.subplots(figsize=(8, 4))
    print(len(train_y))
    y_test_scaled = scaler.inverse_transform(test_y.reshape(-1, 1))
    ax.plot(y_test_scaled, color=(0, 0, 0.7, 0.5), label="True Price")
    plt.plot(predictions, color=(1, 0, 0,0.5),
            label='Predicted Testing Price')
    plt.legend()

    fig, ax2 = plt.subplots(figsize=(8, 4))
    plt.plot(df[ticker].values, color='red', label="True Price")
    ax2.plot(range(len(train_y) + 50, len(train_y) + 50 + len(predictions)), predictions, color='blue',
             label='Predicted Testing Price')
    plt.legend()

def predictFuture(ticker):
    #Create the dataset with the given ticker
    x_test,y_test,x_train,y_train,df,ticker,scaler,values=create_dataset(ticker)

    #Train the RNN network
    train(x_train,y_train,ticker=ticker)

    #Show results of the training
    show_results(x_test,y_test, y_train,ticker,scaler,values,df)

def predictNextYear(ticker):
    # Read data
    df = pd.read_csv('sp500_closes.csv')
    tickers = df.columns.values.tolist()
    # Get Adj Close values of ticker and convert it to numpy array
    values = df[ticker].values
    values = values.reshape(-1, 1)
    # Normalize dataset in range [0-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(values)
    # Create the X and Y variables of the dataset in a format numpy array
    x_train, y_train = create_d_dataset(dataset_train, 50)
    # Prepare data for the neural network
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    train(x_train,y_train,ticker)



predictFuture('AMZN')
