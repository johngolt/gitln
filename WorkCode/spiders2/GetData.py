# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:39:58 2018

@author: Administrator
"""
# 从郑州期货交易所爬取数据。

'''
数据中得到每年各个期货品种的历史数据url链接，并以年来划分。建立字典存储这些
链接。
'''
def get_url(url):
    import requests
    from bs4 import BeautifulSoup
    from collections import defaultdict
    path = r'http://www.dce.com.cn'
    response = requests.get(url)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text,'lxml')
    result = soup.select('.cate_sel.clearfix')# 查找存储数据的地址所在位置
    label = soup.select('option')# 查找年份所在位置
    url_dict = defaultdict(list)
    for value, key in zip(result,label):
        urls = value.select('input')# 得到数据的url地址
        year = key.get('value')# 得到数据年份
        url_dict[year].extend([path+u.get('rel') for u in urls])
        # 存储数据地址链接
    return url_dict

def url_download(url_dict):
    import os
    os.chdir('F:/transactiondata/')
    if not os.path.exists('decdata'):
        os.mkdir('decdata')# 建立本地存储数据的文件夹
    os.chdir('./decdata/')
    import requests
    for year,data in url_dict.items():
        if not os.path.exists('{0}'.format(year)):
            os.mkdir(year)
        os.chdir('./{0}'.format(year))
        for each in data:
            response = requests.get(each)
            path = each.split('/')[-1]# 建立存储数据的本地文件。
            with open(path,'wb') as file:
                file.write(response.content)
        os.chdir('F:/transactiondata/decdata/')
    print('finish')
'''
url= r'http://www.dce.com.cn/dalianshangpin/xqsj/lssj/index.html'
url_dict = get_url(url)
url_download(url_dict)'''


# 读取csv文件，设置编码方式
def read_csv(path,encoding=None):
    import pandas as pd
    import csv
    if encoding:
        with open(path) as file:
            reader = csv.reader(file)
            columns = next(reader)
            data =[]
            for each in reader:
                data.append(each)
    else:
         with open(path,encoding=encoding) as file:
            reader = csv.reader(file)
            columns = next(reader)
            data =[]
            for each in reader:
                data.append(each)
    df = pd.DataFrame(data,columns=columns)
    return df
# 将df中数据类型尝试转换为int,float.如果无法转化，则维持object类型。
def changeType(df):
    columns = df.columns
    for each in columns:
         if all(df[each].str.contains('\.')):
        # 如果数据中含有'.'
            try:
                df[each]=df[each].map(float)
               # 则尝试转换为float类型，如果报错，则不进行转换。
            except:
                continue
         else:
            try:
                df[each]=df[each].map(int)
                # 如果不含有点，则尝试转换为int类型，如果报错，则不进行转换。
            except:
                continue
    return df
# 从转化后的数据中得到主力合约的相关价格指标。
def GetChiefPrice(path,encoding=None):
    df = read_csv(path,encoding)
    df = changeType(df)
    df.drop(['ROWNUM'],axis=1,inplace=True)# 去除计数行
    df['日期']=pd.to_datetime(df['日期'].astype(str),format='%Y%m%d')# 将日期转换为时间形式。
    df = df.set_index(['日期','合约'])# 将日期和合约设为index
    index = df.index.levels[0]
    data = pd.DataFrame(None,columns=df.columns,index = index)
    # 找出每日的最大持仓量对应的合约，作为主力合约，并记录下主力合约相关的价格指标。
    for each in index:
        argindex = df.loc[each,:]['持仓量'].argmax()
        temp = df.loc[each,:]
        data.loc[each,:]=temp.loc[argindex,:]
        data.loc[each,'合约']=argindex
    return data
 
    
    
    
    
    
    
    
# 处理得到的数据，从中得到主力合约
import pandas as pd
import csv
import os
os.chdir('F:/transactiondata/decdata/2016/b')
# 设置工作目录

# 用csv读取csv文件，并存储在列表中
with open('b.csv') as file:
    reader =  csv.reader(file)
    columns=next(reader)# 存储columns
    data = []# 存储数据
    for each in reader:
        data.append(each)
        
df1 = pd.DataFrame(data,columns=columns)
# 利用数据建立DataFrame


df=df1.copy()
# 对表中数据转换形式,转为int,和float型
for each in columns:
    if all(df[each].str.contains('\.')):
        # 如果数据中含有'.'
        try:
            df[each]=df[each].map(float)
           # 则尝试转换为float类型，如果报错，则不进行转换。
        except:
            continue
    else:
        try:
            df[each]=df[each].map(int)
            # 如果不含有点，则尝试转换为int类型，如果报错，则不进行转换。
        except:
            continue

df.drop(['ROWNUM'],axis=1,inplace=True)# 去除计数行
df['日期']=pd.to_datetime(df['日期'].astype(str),format='%Y%m%d')# 将日期转换为时间形式。
df = df.set_index(['日期','合约'])# 将日期和合约设为index
index = df.index.levels[0]
data = pd.DataFrame(None,columns=df.columns,index = index)
# 找出每日的最大持仓量对应的合约，作为主力合约，并记录下主力合约相关的价格指标。
for each in index:
    argindex = df.loc[each,:]['持仓量'].argmax()
    temp = df.loc[each,:]
    data.loc[each,:]=temp.loc[argindex,:]
    data.loc[each,'合约']=argindex    
    
    
