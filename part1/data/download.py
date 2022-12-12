import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np


def insert(df, i, df_add):
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    result = pd.concat([df1,df_add,df2],ignore_index=True)
    return result


def data_download(index_name):
    index=pd.read_csv('./' + index_name + '.csv',header =None)
    
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
    
def process(index_name):
    INDEX_data = pd.read_csv('./'+ str(index_name) +'_data.csv')
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

        count = 0
        for i in holiday_INDEX:
            index = INDEX_data[(INDEX_data['datetime'] < i) &(INDEX_data['ticker']==ticker)].index.tolist()[-1]
            df_add = pd.DataFrame({'Date':[i.strftime('%Y-%m-%d')],'Open':[np.nan],'High':[np.nan],'Low':[np.nan],'Close':[np.nan],'Adj Close':[np.nan],'Volume':[np.nan],'ticker':[ticker],'datetime':[i]})
            INDEX_data = insert(INDEX_data, index+1, df_add)
            count += 1
        print('number of holiday days for '+ str(ticker) + ' in ' + str(index_name)+ ' index : ' + str(len(holiday_INDEX)))                
        assert len(holiday_INDEX) == count
    INDEX_data.to_csv('./'+ str(index_name) +'_data_withmissing'+'.csv')
    
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
    print('generating '+str(len(mask))+' patterns for index '+str(index_name))   
    np.save(str(index_name)+'_mask.npy',np.array(mask))
    
    

    
    
    

    
    
if __name__ == "__main__":
    for index_name in ['DJ30','EU50','HSI']:
        data_download(index_name)
    for index_name in ['DJ30','EU50','HSI']:
        process(index_name)
        
