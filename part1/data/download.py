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
    index=pd.read_csv('./datasets/'+index_name+'/' + index_name + '.csv',header =None)
    
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

    all_data.to_csv('./datasets/'+index_name+'/'+index_name+'_data.csv')
    print('Finished downloaded data for '+index_name+' and save in '+index_name+'_data.csv')

    

def process(index_name):
    """
    process stock data to generate model input array, and the historical mask pattern

    Parameters:
    index_name(str):                choose from ['HSI','DJ30','EU50']
    """
    
    # insert the missing values in the downloaded dataframe 
    INDEX_data = pd.read_csv('./datasets/'+index_name+'/'+index_name+'_data.csv')
    tickers = list(set(INDEX_data['ticker'].tolist()))
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
    #INDEX_data.to_csv('./'+ str(index_name) +'_data_withmissing'+'.csv')
    
    if index_name == 'EU50':
        holiday_INDEX = holiday_INDEX_DE
    
    #generate mask patterns according to historical patterns
    mask = []
    count = 0
    datetime_series = INDEX_data['datetime'].tolist()
    month = 0
    while datetime.strptime('2012-12-03', '%Y-%m-%d') + timedelta(days=month*28) < datetime_series[-1]:
        start = datetime.strptime('2012-12-03', '%Y-%m-%d') + timedelta(days=month*28)
        mask_month = []
        for i in range(28):
            if start + timedelta(days=i) not in holiday_INDEX and start + timedelta(days=i) in INDEX_data['datetime'].tolist():
                mask_month.append(0)
            elif start + timedelta(days=i) in holiday_INDEX and start + timedelta(days=i) in INDEX_data['datetime'].tolist():
                mask_month.append(1)
                count += 1
            else:
                continue
        mask.append(mask_month)
        month += 1

    assert count == len(holiday_INDEX) 

    for add_zero in range(20 - len(mask[-1])):
        mask[-1].append(0)
        
    mask = np.expand_dims(mask,axis=-1)
    mask = np.repeat(mask,6,axis=2)
    mask = np.unique(mask,axis=0)

    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            index = i
    mask = np.delete(mask,index, 0)
    mask = np.where(mask==0,1,0)
    
    print('generating '+str(len(mask))+' patterns for index '+str(index_name))   
    np.save('./datasets/'+index_name+'/'+index_name+'_mask.npy',np.array(mask))
   
    
    # data normalization 
    #INDEX_data = INDEX_data.drop(columns=['Unnamed: 0'])
    ticker_list = list(set(INDEX_data['ticker'].tolist()))
    
    INDEX_data_normed = pd.DataFrame({'Date':[],'Open':[],'High':[],'Low':[],'Close':[],'Adj Close':[],'Volume':[],'ticker':[],'datetime':[]})
    for ticker in ticker_list:
        subdf = INDEX_data[INDEX_data['ticker'] == ticker].reset_index()
        subdf = subdf.drop(columns=['index'])
        test = pd.DataFrame({'Date':subdf['Date'],'Open':preprocess(subdf['Open']),'High':preprocess(subdf['High']),
                             'Low':preprocess(subdf['Low']),'Close':preprocess(subdf['Close']),
                             'Adj Close':preprocess(subdf['Adj Close']), 'Volume':preprocess(subdf['Volume']),
                             'ticker':subdf['ticker'],'datetime':subdf['datetime']})
        INDEX_data_normed = pd.concat([INDEX_data_normed,test])
        
    
    #generate training/validation/test array
    tickers = list(set(INDEX_data_normed['ticker'].to_list()))

    
    single = INDEX_data_normed[INDEX_data_normed['ticker']==tickers[0]].reset_index()
    single = single.drop(columns=['index'])
    total_length = single.shape[0]
    train_length = int(total_length*0.7)
    val_length = int(total_length*0.1)
    test_length = int(total_length*0.2)
    
    
    train_data = single.loc[0:20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
    for i in range(1,int(train_length//10)-1):
        start_index = i*10
        train_data = np.vstack((train_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))
        
    val_data = single.loc[(int(train_length//10)-1)*10:(int(train_length//10)-1)*10+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
    for i in range(int(train_length//10),int((train_length+val_length)//10)-1):
        start_index = i*10
        val_data = np.vstack((val_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))
        
    test_data = single.loc[(int((train_length+val_length)//10)-1)*10:(int((train_length+val_length)//10)-1)*10+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
    for i in range(int((train_length+val_length)//10),int((train_length+val_length+test_length)//10)-1):
        start_index = i*10
        test_data = np.vstack((test_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))
    
    total_train = train_data
    total_val = val_data
    total_test = test_data


    for ticker in tickers[1:]:
        single = INDEX_data_normed[INDEX_data_normed['ticker']==ticker].reset_index()
        single = single.drop(columns=['index'])
        total_length = single.shape[0]
        
        train_length = int(total_length*0.7)
        val_length = int(total_length*0.1)
        test_length = int(total_length*0.2)

        train_data = single.loc[0:20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
        for i in range(1,int(train_length//10)-1):
            start_index = i*10
            train_data = np.vstack((train_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))

        val_data = single.loc[(int(train_length//10)-1)*10:(int(train_length//10)-1)*10+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
        for i in range(int(train_length//10),int((train_length+val_length)//10)-1):
            start_index = i*10
            val_data = np.vstack((val_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))

        test_data = single.loc[(int((train_length+val_length)//10)-1)*10:(int((train_length+val_length)//10)-1)*10+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()
        for i in range(int((train_length+val_length)//10),int((train_length+val_length+test_length)//10)-1):
            start_index = i*10
            test_data = np.vstack((test_data,single.loc[start_index:start_index+20-1,['Open','High','Low','Close','Adj Close','Volume']].to_numpy()))

        total_train = np.vstack((total_train,train_data))
        total_val = np.vstack((total_val,val_data))
        total_test = np.vstack((total_test,test_data))

    
    training_data = np.reshape(total_train,[-1,20,6])
    validation_data = np.reshape(total_val,[-1,20,6])
    testing_data = np.reshape(total_test,[-1,20,6])
  

    np.save('./datasets/'+index_name+'/'+index_name+'_train.npy',training_data)
    np.save('./datasets/'+index_name+'/'+index_name+'_val.npy',validation_data)
    np.save('./datasets/'+index_name+'/'+index_name+'_test.npy',testing_data)

    

if __name__ == "__main__":
    for index_name in ['DJ30','EU50','HSI']:
        data_download(index_name)
    for index_name in ['DJ30','EU50','HSI']:
        process(index_name)

