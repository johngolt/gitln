# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:07:25 2018

@author: Administrator
"""
import pandas as pd
import os

'''
根据需要对库存表和价格序列进行调整，并得到用于后续处理
的表格，表格中包含：库存因子，收盘价，开盘价，当前持仓量最大合约名，
price:为期货相关的价格信息，包括：开盘价，最高，最低价，收盘价，持仓量最大合约名，
日期。
inventory：库存相关的信息，包括：date,期货wind代码，期货名
'''
os.chdir(r'F:/transactiondata/inventory/')  
def changedf(price,inventory,N=5):
    ##从价格序列中提取需要的特征
    use_price = price[['日期','开盘价','收盘价','name']]
    #转换为日期格式。
    inventory['date']=pd.to_datetime(inventory['date'],format='%Y%m%d')
    use_price['日期']=pd.to_datetime(use_price['日期'],format = '%Y-%m-%d')
    #根据价格序列日期，调整库存的时间。
    index =inventory['date']>= use_price['日期'][0]
    invent = inventory.loc[index,:]
    #根据库存计算库存因子。
    invent.loc[:,'mean']=invent.iloc[:,1].rolling(window=N).mean()
    invent.loc[:,'factor']=invent.iloc[:,1]/invent.loc[:,'mean'].shift(1)-1
    invent = invent.set_index('date')
    use_price= use_price.set_index('日期')
    #合并库存序列和价格序列。
    total = pd.merge(invent,use_price,left_index=True,right_index=True)
    total['factor']=total['factor'].shift(1)
    total['name1']=total.name['name'].shift(-1)
    total = total[['factor','开盘价','收盘价','name','name1']].dropna()
  #  total = total[['factor','开盘价','收盘价','name']].dropna()
    return total

    
'''根据库存因子确定做多和做空点，并进行交易'''
def BuySell1(series):
    global money,flag,stock,remain, times
    if flag==0:# 如果现在处于没有操作的情况下
        if series['factor']>0:# 如果库存因子大于0,做空期货
            stock -= money//series['开盘价']# 做空期货的数目
            remain = money-stock*series['开盘价']*(1-fee)# 做空期货后，手中的钱
            money = remain+stock*series['收盘价']# 当天收盘时的资金额
            flag=-1
            times+=1
            return money
        else:
            stock += money//series['开盘价'] #做多期货
            remain = money-stock*series['开盘价']*(1+fee)# 做多期货后手中剩余的钱
            money= remain+stock*series['收盘价']#实际手中的钱
            flag=1
            times+=1
            return money
    elif flag==1:# 做多情况下
        if series['factor']>0:# 库存因子为正，做空期货
            money = remain+stock*series['开盘价']*(1-fee)# 买完期权之后的现金
            stock=0# 将手中期权清空
            stock -= money//series['开盘价']# 做空期货的数目
            remain = money-stock*series['开盘价']*(1-fee)# 做空期货后，手中的钱
            money = remain+stock*series['收盘价']
            flag=-1
            times+=1
            return money
        else:
            if series['name']==series['name1']:
                money = remain+stock*series['收盘价']
            else:
                money = remain+stock*series['收盘价']*(1-rollcost)
            return money
    else:
        if series['factor']<0:
            money = remain+stock*series['开盘价']*(1+fee) # 平仓之后的现金
            stock=0
            stock+=money//series['开盘价'] #做多期货
            remain = money-stock*series['开盘价']*(1+fee) # 做多期货后手中剩余的钱
            money= remain+stock*series['收盘价'] # 实际手中的钱
            flag=1
            times+=1
            return money
        else:
            if series['name']==series['name1']:
                money = remain+stock*series['收盘价']
            else:
                money = remain+stock*series['收盘价']*(1+rollcost)
            return money 
    
'''计算最大回测率'''
def Maximun_back_test(money):
    diff = money - money.shift(1) # 计算价差
    current,total = 0,0 # 记录当前最小值和全局最小值 
    start, end = 0, 0 # 记录当前全局最小值的起始点和终点
    for i,item in enumerate(diff[1:],start = 1):
        if current+item>=item: # 如果当前点之前的点和大于0，则记录下当前点坐标
            index = i
        current = min(current+item,item)# 记录下当前最小值
        if total>current:#如果当前的最小值，大于全局最小值，则记录下当前最小值的起点和终点。
            end = i
            start = index
        total = min(total,current)# 记录下全局最小值
    rate = (money[end]-money[start-1])/money[start-1] # 计算最大回测率
    return rate
    
'''计算年化收益率，标准差，夏普比率。'''
def sharp_ratio(money):
    import numpy as np
    n = len(money)
    rate = money.iloc[-1]/money.iloc[0]
    rate = pow(rate,1/n)
    returns = rate**252# 年化收益率
    diff = money/money.shift(1)-1#日收益率
    diff = diff.dropna()
    std = diff.std()*np.sqrt(252)# 日收益率的年化标准差
    sharp = returns/std # 夏普比率
    return [returns,std,sharp]
'''    
fee = 0.0003# 手续费
rollcost=fee*(2-fee) #展期费用
money = 1000000# 初始资金
remain = 0#手中剩余资金
flag = 0# 做多做空状态
stock=0# 初始期权数目'''
os.chdir('F:/transactiondata/inventory/')
inventories = os.listdir()
inventories = [dic for dic in inventories if dic.endswith('csv')]
prices = os.listdir('./price/')
year =[5]+list(range(10,100,10))
name = [item.split('.')[0] for item in inventories]
index = pd.MultiIndex.from_product([year,name])
df = pd.DataFrame(None,index = index,columns=['mean','std','sharp'])

for N in year:
    ls=[]
    for path1,path2 in  zip(inventories,prices):
        inventory = pd.read_csv(path1)
        price = pd.read_csv('./price/'+path2,encoding='gbk')
        total = changedf(price,inventory,N)
        fee = 0.0003# 手续费
        rollcost=fee*(2-fee) #展期费用
        money = 1000000# 初始资金
        remain = 0#手中剩余资金
        flag = 0# 做多做空状态
        stock=0# 初始期权数目
        times = 0
        money =  total.apply(BuySell1,axis=1)
        sharp = sharp_ratio(money[:'2018-01-26'])
        max_back= Maximun_back_test(money[:'2018-01-16'])
        sharp.append(max_back)
        ls.append(sharp)
    df.loc[N,:]=ls