import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from sklearn import preprocessing
import warnings 
warnings.filterwarnings('ignore')


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


def data_download(index_name):
    """
    Download stock data through yfinance 
    Parameters:
    index_name(str):       choose from ['HSI','DJ30','EU50']
    """
    index=pd.read_csv(index_name + '.csv',header =None)
    
    if index_name == 'HSI':
        index0 = list(index.iloc[:,0])
        index = [str(i).zfill(4)+'.HK' for i in index0]
    else:
        index=list(index.iloc[:,0])
        
    all_data = yf.download(index[0], start="2012-12-01", end="2022-11-30").reset_index()
    all_data.insert(all_data.shape[1],'ticker', index[0])

    for ticker in index[1:]:
        data = yf.download(ticker, start="2012-12-01", end="2022-11-30").reset_index()
        try:
            if  data.loc[0,'Date'] > datetime.strptime("2012-12-03", '%Y-%m-%d'):
                print(ticker + ' does not have at least 10 years of history, please double check')
            else:
                data.insert(data.shape[1], 'ticker', ticker)
                objs = [all_data,data]
                all_data = pd.concat(objs, axis=0)
        except:
            print('Data download for '+ticker+' fails, please double check')

    all_data.to_csv(index_name+'_data.csv')
    print('Finished downloaded data for '+index_name+' and save in '+index_name+'_data.csv')

    

def generate(index_name,data_path,time_stamp):
    ############################################################################
    #               generate observation  with missing value                   #
    ############################################################################


    # insert the missing values in the downloaded dataframe 
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



    ##########################################################################################
    #          generate dataframe with column indexs == tickers, row index == date           #
    ##########################################################################################

    test_time = ['2016-12-01','2017-11-30','2021-11-30','2022-11-30']
    val_time = ['2018-12-01','2019-11-30']

    tickers = list(set((INDEX_data['ticker'].tolist())))
    df_adjclose = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        df_adjclose[ticker] = (INDEX_data[INDEX_data['ticker']==ticker]['Adj Close']).to_numpy()

    df_adjclose['Date'] = INDEX_data['Date'].tolist()[: df_adjclose.shape[0]]   
    df_adjclose = df_adjclose.reset_index()
    df_adjclose = df_adjclose.drop(columns = ['index'])
    df_adjclose['Date'] = pd.to_datetime(df_adjclose['Date'])

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

    with open(data_path +'/'+index_name+'_mean_std.pickle', 'wb') as handle:
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
    np.save(data_path+'/'+index_name+'_train.npy',train)


    for i in range(val1.shape[0] - time_stamp + 1):
        if i == 0:
            val = val1.loc[i:i+time_stamp-1,tickers].to_numpy()
        else:
            val = np.concatenate([val,val1.loc[i:i+time_stamp-1,tickers].to_numpy()])   
    val = np.reshape(val,[-1,time_stamp,num_of_tickers])   
    np.save(data_path+'/'+index_name+'_val.npy',val)

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
    np.save(data_path+'/'+index_name+'_test.npy',test)

    ###########################################################################################
    # generate mask                                                                           #
    ###########################################################################################

    for i in range(train1.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        if i == 0:
            mask = np.where(train1.loc[start:end-1,tickers].to_numpy(),1,0)
        else:
            mask = np.concatenate([mask,np.where(np.isnan(train1.loc[start:end-1,tickers].to_numpy()),0,1)]) 

    for i in range(train2.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        mask = np.concatenate([mask,np.where(np.isnan(train2.loc[start:end-1,tickers].to_numpy()),0,1)]) 

    for i in range(val1.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        mask = np.concatenate([mask,np.where(np.isnan(val1.loc[start:end-1,tickers].to_numpy()),0,1)]) 

    for i in range(train3.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        mask = np.concatenate([mask,np.where(np.isnan(train3.loc[start:end-1,tickers].to_numpy()),0,1)]) 

    mask = np.reshape(mask,[-1,time_stamp,num_of_tickers])   
    mask = np.unique(mask,axis=0)

    index_delete = []
    for i in range(mask.shape[0]):
        if mask[i].sum() == (time_stamp*num_of_tickers):
            index_delete.append(i)
    if len(index_delete)!= 0:
        mask = np.delete(mask,index_delete, 0)



    np.save(data_path+'/'+index_name+'_mask_train.npy',mask)


    for i in range(test1.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        if i == 0:
            mask_test = np.where(np.isnan(test1.loc[start:end-1,tickers].to_numpy()),0,1)
            missing_index = list(np.where(np.sum(mask_test,axis=1)<num_of_tickers)[0])
            non_missing = [j for j in range(time_stamp) if j not in missing_index]
            replace_index = random.sample(non_missing,len(missing_index))
            mask_test[replace_index,:] = mask_test[missing_index,:]
            mask_test_all = mask_test

        else:
            mask_test = np.where(np.isnan(test1.loc[start:end-1,tickers].to_numpy()),0,1)
            missing_index = list(np.where(np.sum(mask_test,axis=1)<num_of_tickers)[0])
            non_missing = [j for j in range(time_stamp) if j not in missing_index]
            replace_index = random.sample(non_missing,len(missing_index))
            mask_test[replace_index,:] = mask_test[missing_index,:]

            mask_test_all = np.concatenate([mask_test_all,mask_test]) 

    for i in range(test2.shape[0]//time_stamp):
        start = i*time_stamp
        end = (i+1)*time_stamp
        mask_test = np.where(np.isnan(test2.loc[start:end-1,tickers].to_numpy()),0,1)
        missing_index = list(np.where(np.sum(mask_test,axis=1)<num_of_tickers)[0])
        non_missing = [j for j in range(time_stamp) if j not in missing_index]
        replace_index = random.sample(non_missing,len(missing_index))
        mask_test[replace_index,:] = mask_test[missing_index,:]

        mask_test_all = np.concatenate([mask_test_all,mask_test]) 


    mask_test_all=np.reshape(mask_test_all,[-1,time_stamp,num_of_tickers])        
    np.save(data_path+'/'+index_name+'_mask_test.npy',mask_test_all)

    

if __name__ == "__main__":
    for index_name in ['DJ30','EU50','HSI']:
        data_download(index_name)
    for index_name in ['DJ30','EU50','HSI']:
        if not os.path.exists('./datasets/'+index_name):
            os.makedirs('./datasets/'+index_name) 
        generate(index_name = index_name,data_path='./datasets/'+index_name,time_stamp=30)
