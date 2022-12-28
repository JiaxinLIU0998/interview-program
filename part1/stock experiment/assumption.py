from datetime import timedelta,datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def assumption_test(index_name,pre_or_post):
    
    INDEX_data = pd.read_csv(index_name+'_data.csv')
    tickers = list(set(INDEX_data['ticker'].tolist()))
    INDEX_data = INDEX_data.drop(columns=['Unnamed: 0'])
    delta = timedelta(days=1)

    datetime_list = []
    for i in range(INDEX_data.shape[0]):
        datetime_list.append(datetime.strptime(INDEX_data['Date'][i], '%Y-%m-%d'))
    INDEX_data['datetime'] = pd.Series(datetime_list)

    all_df = pd.DataFrame(columns=['ticker','return_series','dummy'])
    #all_df = pd.DataFrame({'ticker':[],'price':[],'dummy':[]})
    for ticker in tickers:
        single_df = INDEX_data[INDEX_data['ticker'] == ticker].reset_index()
        INDEX_time_stamp = single_df['datetime'].tolist()
        start = INDEX_time_stamp[0]
        end = INDEX_time_stamp[-1]

        holiday_day = []
        while start < end:
            if start not in INDEX_time_stamp and int(start.weekday()) != 5 and int(start.weekday()) != 6:  
                if pre_or_post == 'pre':
                    holiday_day.append(start - delta)
                if pre_or_post == 'post':
                    holiday_day.append(start + delta)
            start = start + delta

        holiday_day = list(set(holiday_day).intersection(set(INDEX_time_stamp)))
        dummy = [1 if single_df.loc[i,'datetime'] in holiday_day else 0 for i in range(single_df.shape[0]) ]

        return_series = []
        price_list = single_df['Adj Close'].tolist()
        for i in range(1,len(price_list)):
            return_series.append((price_list[i] - price_list[i-1])/price_list[i-1])


        ticker_df = pd.DataFrame({'ticker':pd.Series(single_df['ticker'].tolist()[1:]),
                                  'return_series':pd.Series(return_series),
                                  'dummy':pd.Series(dummy[1:])})
        all_df = pd.concat([all_df,ticker_df])
    
    
    X=np.array(all_df['dummy'].to_numpy().reshape(-1, 1), dtype=float)
    X2 = sm.add_constant(X)
    est = sm.OLS(all_df['return_series'].to_numpy(), X2)
    est2 = est.fit()
    print('for index:')
    print(est2.pvalues[1])
    
    
    tickers = list(set(all_df['ticker'].tolist()))
    for ticker in tickers:
        single_df = all_df[all_df['ticker'] == ticker].reset_index()
        X=np.array(single_df['dummy'].to_numpy().reshape(-1, 1), dtype=float)
        X2 = sm.add_constant(X)
        est = sm.OLS(single_df['return_series'].to_numpy(), X2)
        est2 = est.fit()
        print(ticker)
        print(est2.pvalues[1])

