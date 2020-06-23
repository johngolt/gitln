import pandas as pd
import sklearn as skr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import OneHotEncoder

# Load the balance data
def load_data(path='user_balance_table.csv'):
    data_balance = pd.read_csv(path)
    data_balance = add_timestamp(data_balance)
    return data_balance.reset_index(drop=True)
    

# add tiemstamp to dataset
def add_timestamp(data, time_index='report_date'):
    data_balance = data.copy()
    data_balance['date'] = pd.to_datetime(data_balance[time_index],
                                          format= "%Y%m%d")
    data_balance['day'] = data_balance['date'].dt.day
    data_balance['month'] = data_balance['date'].dt.month
    data_balance['year'] = data_balance['date'].dt.year
    data_balance['week'] = data_balance['date'].dt.week
    data_balance['weekday'] = data_balance['date'].dt.weekday
    return data_balance.reset_index(drop=True)

# total amount
def get_total_balance(data,date):
    df_tmp = data.copy()
    df_tmp = df_tmp.groupby(['date'])['total_purchase_amt',
                                      'total_redeem_amt'].sum()
    df_tmp.reset_index(inplace=True)
    return df_tmp[(df_tmp['date']>= date)].reset_index(drop=True)

# Generate the test data
def generate_test_data(data):
    total_balance = data.copy()
    start = datetime.datetime(2014,9,1)
    testdata = []
    while start != datetime.datetime(2014,10,15):
        temp = [start, np.nan, np.nan]
        testdata.append(temp)
        start += datetime.timedelta(days = 1)
    testdata = pd.DataFrame(testdata)
    testdata.columns = total_balance.columns

    total_balance = pd.concat([total_balance, testdata], axis = 0)
    total_balance = total_balance.reset_index(drop=True)
    return total_balance.reset_index(drop=True)

# 定义生成时间序列规则预测结果的方法
def generate_base(df, month_index):
    # 选中固定时间段的数据集
    total_balance = df.copy()
    total_balance = total_balance[['date', 'total_purchase_amt', 'total_redeem_amt']]
    total_balance = total_balance[(total_balance[
        'date'] >= pd.Timestamp(datetime.date(2014,3,1))) & (total_balance[
        'date'] < pd.Timestamp(datetime.date(2014, month_index, 1)))]

    # 加入时间戳
    total_balance['weekday'] = total_balance['date'].dt.weekday
    total_balance['day'] = total_balance['date'].dt.day
    total_balance['week'] = total_balance['date'].dt.week
    total_balance['month'] = total_balance['date'].dt.month
    
    # 统计翌日因子
    mean_of_each_weekday = total_balance[
        ['weekday']+['total_purchase_amt','total_redeem_amt']
    ].groupby('weekday',as_index=False).mean()
    for name in ['total_purchase_amt','total_redeem_amt']:
        mean_of_each_weekday = mean_of_each_weekday.rename(
            columns={name: name+'_weekdaymean'})

    mean_of_each_weekday['total_purchase_amt_weekdaymean'] /= np.mean(
        total_balance['total_purchase_amt'])

    mean_of_each_weekday['total_redeem_amt_weekdaymean'] /= np.mean(
        total_balance['total_redeem_amt'])

    # 合并统计结果到原数据集
    total_balance = pd.merge(total_balance, 
                             mean_of_each_weekday, on='weekday', how='left')

    # 分别统计翌日在(1~31)号出现的频次
    weekday_count = total_balance[['day','weekday','date']
                                 ].groupby(['day','weekday'],as_index=False).count()
    weekday_count = pd.merge(weekday_count, 
                             mean_of_each_weekday, on='weekday')

    # 依据频次对翌日因子进行加权，获得日期因子
    weekday_count['total_purchase_amt_weekdaymean'] *= weekday_count[
        'date']   / len(np.unique(total_balance['month']))
    weekday_count['total_redeem_amt_weekdaymean'] *= weekday_count[
        'date']  / len(np.unique(total_balance['month']))

    day_rate = weekday_count.drop(
        ['weekday','date'],axis=1).groupby('day',as_index=False).sum()

    # 将训练集中所有日期的均值剔除日期残差得到base
    day_mean = total_balance[
        ['day'] + ['total_purchase_amt','total_redeem_amt']
                         ].groupby('day',as_index=False).mean()
    day_pre = pd.merge(day_mean, day_rate, on='day', how='left')

    day_pre['total_purchase_amt'] /= day_pre[
        'total_purchase_amt_weekdaymean']
    day_pre['total_redeem_amt'] /= day_pre['total_redeem_amt_weekdaymean']

    # 生成测试集数据
    for index, row in day_pre.iterrows():
        if month_index in (2,4,6,9) and row['day'] == 31:
            break
        day_pre.loc[index, 'date'] = datetime.datetime(
            2014, month_index, int(row['day']))

    # 基于base与翌日因子获得最后的预测结果
    day_pre['weekday'] = day_pre.date.dt.weekday
    day_pre = day_pre[['date','weekday']+['total_purchase_amt','total_redeem_amt']]

    day_pre = pd.merge(day_pre, mean_of_each_weekday,on='weekday')

    day_pre['total_purchase_amt'] *= day_pre['total_purchase_amt_weekdaymean']
    day_pre['total_redeem_amt'] *= day_pre['total_redeem_amt_weekdaymean']

    day_pre = day_pre.sort_values(
        'date')[['date']+['total_purchase_amt','total_redeem_amt']]
    return day_pre