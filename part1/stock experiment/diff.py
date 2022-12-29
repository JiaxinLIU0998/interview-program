import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from sklearn import preprocessing
import warnings 
import random
import pickle
warnings.filterwarnings('ignore')
import os
import sys


def insert(df, i, df_add):
    """
    Insert one row in dataframe

    Inputs:
    df:                       Dataframe    
    i:                        insert position
    df_add:                   insert content
    
    Returns:                  Dataframe                
    """
    
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    result = pd.concat([df1,df_add,df2],ignore_index=True)
    return result





def preprocess(x):
    """
    Normalization for features of each stock

    Inputs:
    x:                        Series
    
    Returns:                  Series
    """
    return pd.Series(preprocessing.scale(np.array(x), axis=0, with_mean=True,with_std=True,copy=True))


def generate(index_name):
    ############################################################################
    #               generate observation  with missing value                   #
    ############################################################################


    # insert the missing values in the downloaded dataframe 
    #INDEX_data = pd.read_csv('/home/root/jpm/SSSD/tf/stock_data/'+index_name+'_data.csv')
    INDEX_data = pd.read_csv(index_name+'_data.csv')
    tickers = list(set(INDEX_data['ticker'].tolist()))
    num_of_tickers = len(tickers)
    INDEX_data = INDEX_data.drop(columns=['Unnamed: 0'])
    delta = timedelta(days=1)

    datetime_list = []
    for i in range(INDEX_data.shape[0]):
        datetime_list.append(datetime.strptime(INDEX_data['Date'][i], '%Y-%m-%d'))
    INDEX_data['datetime'] = pd.Series(datetime_list)

    for ticker in tickers:
        INDEX_time_stamp = INDEX_data[INDEX_data['ticker'] == ticker]['datetime'].tolist()
        start = INDEX_time_stamp[0]
        end = INDEX_time_stamp[-1]

        holiday_INDEX = []
        while start < end:
            if start not in INDEX_time_stamp and int(start.weekday()) != 5 and int(start.weekday()) != 6:  
                holiday_INDEX.append(start)
            start = start + delta

        if ticker[-2:] == 'DE':
            holiday_INDEX_DE = holiday_INDEX

        count = 0
        for i in holiday_INDEX:
            index = INDEX_data[(INDEX_data['datetime'] < i) &(INDEX_data['ticker']==ticker)].index.tolist()[-1]
            df_add = pd.DataFrame({'Date':[i.strftime('%Y-%m-%d')],'Open':[np.nan],'High':[np.nan],'Low':[np.nan],'Close':[np.nan],'Adj Close':[np.nan],'Volume':[np.nan],'ticker':[ticker],'datetime':[i]})
            INDEX_data = insert(INDEX_data, index+1, df_add)
            count += 1
        print('number of holiday days for '+ str(ticker) + ' in ' + str(index_name)+ ' index : ' + str(len(holiday_INDEX)))                
        assert len(holiday_INDEX) == count
    
    
    tickers = list(set((INDEX_data['ticker'].tolist())))
    df_adjclose = pd.DataFrame(columns=tickers)
    num_of_dates = INDEX_data[INDEX_data['ticker'] == tickers[0]].reset_index().shape[0]
    df_adjclose['datetime'] = INDEX_data['datetime'][:num_of_dates]
    for ticker in tickers:
        df_adjclose[ticker] = (INDEX_data[INDEX_data['ticker']==ticker]['High']).to_numpy() - (INDEX_data[INDEX_data['ticker']==ticker]['Low']).to_numpy()
    return df_adjclose
        
 

if __name__ == "__main__":

    time_stamp = sys.argv[1]
    data_path = './datasets/diff'


    ##########################################################################################
    #          generate dataframe with column indexs == tickers, row index == date           #
    ##########################################################################################

    df_adjclose_DL30 = generate('DJ30')
    df_adjclose_EU50 = generate('EU50')
    df_adjclose_HSI = generate('HSI')

    df_adjclose_EU50 = df_adjclose_EU50.drop(columns = ['datetime'])
    df_adjclose_HSI = df_adjclose_HSI.drop(columns = ['datetime'])
    num_DJ30 = len(df_adjclose_DL30.columns.tolist())-1
    num_EU50 = len(df_adjclose_EU50.columns.tolist())
    num_HSI = len(df_adjclose_HSI.columns.tolist())

    df_adjclose=pd.concat([df_adjclose_EU50,df_adjclose_DL30,df_adjclose_HSI],axis=1)

    df_adjclose = df_adjclose.dropna(axis=0, how='any')



    if not os.path.exists(data_path):
        os.makedirs(data_path) 


    test_time = ['2016-12-01','2017-11-30','2021-11-30','2022-11-30']
    val_time = ['2018-12-01','2019-11-30']
    df_adjclose['Date'] = pd.to_datetime(df_adjclose['datetime'])
    df_adjclose = df_adjclose.drop(columns = ['datetime'])
    tickers = [i for i in df_adjclose.columns.tolist() if i != 'Date']
    num_of_tickers = len(tickers)

    ###########################################################################################
    # use the mean/std of training_validatin set to normalize data                            #
    ###########################################################################################
    train_all = pd.concat([df_adjclose[df_adjclose['Date']<datetime.strptime(test_time[0], '%Y-%m-%d')].reset_index(),
                           df_adjclose[datetime.strptime(test_time[1],'%Y-%m-%d')<df_adjclose['Date']][df_adjclose['Date']<datetime.strptime(test_time[2], '%Y-%m-%d')].reset_index()
                          ])

    mean_std = {}
    for i in train_all.columns.tolist():
        if i != 'index' and i != 'Date':
            mean_std[i] = [np.mean(train_all[i]),np.std(train_all[i])]

    for i in df_adjclose.columns:
        if i in mean_std.keys():
            df_adjclose[i] = (df_adjclose[i] - mean_std[i][0])/mean_std[i][1]

    with open(data_path +'/diff_mean_std.pickle', 'wb') as handle:
        pickle.dump(mean_std, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ###########################################################################################
    # generate training/valisation/testing set (.npy)                                         #
    ###########################################################################################


    train1 = df_adjclose[df_adjclose['Date']<datetime.strptime(test_time[0], '%Y-%m-%d')].reset_index()
    test1 = df_adjclose[datetime.strptime(test_time[0], '%Y-%m-%d')<=df_adjclose['Date']][df_adjclose['Date']<=datetime.strptime(test_time[1],'%Y-%m-%d')].reset_index()
    train2 = df_adjclose[datetime.strptime(test_time[1],'%Y-%m-%d')<df_adjclose['Date']][df_adjclose['Date']<datetime.strptime(val_time[0], '%Y-%m-%d')].reset_index()
    val1 = df_adjclose[datetime.strptime(val_time[0], '%Y-%m-%d')<=df_adjclose['Date']][df_adjclose['Date']<=datetime.strptime(val_time[1], '%Y-%m-%d')].reset_index()
    train3 = df_adjclose[datetime.strptime(val_time[1], '%Y-%m-%d')<df_adjclose['Date']][df_adjclose['Date']<=datetime.strptime(test_time[2], '%Y-%m-%d')].reset_index()
    test2 = df_adjclose[datetime.strptime(test_time[2], '%Y-%m-%d')<df_adjclose['Date']][df_adjclose['Date']<=datetime.strptime(test_time[3], '%Y-%m-%d')].reset_index()


    for i in range(train1.shape[0] - time_stamp + 1):
        if i == 0:
            train = train1.loc[i:i+time_stamp-1,tickers].to_numpy()
        else:
            train = np.concatenate([train,train1.loc[i:i+time_stamp-1,tickers].to_numpy()])       
    for i in range(train2.shape[0] - time_stamp + 1):
        train = np.concatenate([train,train2.loc[i:i+time_stamp-1,tickers].to_numpy()])    
    for i in range(train3.shape[0] - time_stamp + 1):
        train = np.concatenate([train,train3.loc[i:i+time_stamp-1,tickers].to_numpy()])

    train = np.reshape(train,[-1,time_stamp,num_of_tickers])
    np.save(data_path+'/diff_train.npy',train)


    for i in range(val1.shape[0] - time_stamp + 1):
        if i == 0:
            val = val1.loc[i:i+time_stamp-1,tickers].to_numpy()
        else:
            val = np.concatenate([val,val1.loc[i:i+time_stamp-1,tickers].to_numpy()])   
    val = np.reshape(val,[-1,time_stamp,num_of_tickers])   
    np.save(data_path+'/diff_val.npy',val)

    for i in range(test1.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        if i == 0:
            test = test1.loc[start:end-1,tickers].to_numpy()
        else:
            test = np.concatenate([test,test1.loc[start:end-1,tickers].to_numpy()]) 

    for i in range(test2.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        test = np.concatenate([test,test2.loc[start:end-1,tickers].to_numpy()]) 

    test = np.reshape(test,[-1,time_stamp,num_of_tickers]) 
    np.save(data_path+'/diff_test.npy',test)

    ###########################################################################################
    # generate mask                                                                           #
    ###########################################################################################


    single = [1 if i < (num_DJ30+num_EU50) else 0 for i in range(num_DJ30+num_EU50+num_HSI)]
    mask  = np.repeat(np.expand_dims(single,axis=0),time_stamp,axis=0)
    np.save(data_path+'/diff_mask.npy',mask)

    



