def ChIventory(lists, N=5):
    items = []
    #per={'au':1000,'zn':5,'ni':5,'cu':5,'al':5,'pb':5,'sn':1,'ag':15}
    os.chdir('./inventory')
    for item in lists:
        df = pd.read_csv(item)
        df['date']=pd.to_datetime(df['date'],format='%Y-%m-%d')
        df=df.set_index('date')
       # each = per[df.columns[0]]
        name=df.columns[0]
        #df[name+'_per']=df[name]*each 
        mean = df.iloc[:,0].rolling(window=5).mean()
        df.loc[:,name+'_factor']=df.loc[:,name]/mean.shift(1)-1
        df.drop([name],axis=1,inplace=True)
        items.append(df)
    
    return pd.concat(items,axis=1,join='inner')

def ChPrice(lists):
    per={'au':1000,'zn':5,'ni':5,'cu':5,'al':5,'pb':5,'sn':1,'ag':15}
    items=[]
    os.chdir('./price')
    for item in lists:
        name = item.split('.')[0][:2]
        df = pd.read_csv(item,encoding='gbk')
        df['日期']=pd.to_datetime(df['日期'],format = '%Y-%m-%d')
        df.set_index(['日期'],inplace=True)
        df['close']=df['收盘价']*per[name]
        df['open']=df['开盘价']*per[name]
        df['name1']=df['name'].shift(-1)
        df = df[['open','close','name','name1']]
        df.columns=[name+'_'+each for each in df.columns]
        items.append(df)

result = pd.concat([price,inventory],axis=1,join='inner')

def Merge(dfs,week = 1):
    df = pd.concat(dfs,axis=1,join='inner')
    df.index.name='date'
    df.reset_index(inplace=True)
    df['IsMonday']=df['date'].dt.dayofweek==0
    if week !=1:
        df['IsWeek']=(df['IsMonday'].cumsum()%2==1)&(df['IsMonday'])
    else:
        df['IsWeek']=df['IsMonday']
    df.drop(['IsMonday'],axis=1,inplace=True)
    return df
dfs = [price,inventory]
merge = Merge(dfs,week=2)


moneys = 4000000
stock = {}
remains=0
fee=0.0003
rollcost = fee*(2-fee)
flag=0
def BuySells(series):
    global moneys, stock, remains,flag
    if series['IsWeek']:
        if flag==0:  # 当前没有操作的情况下
            factor = [item for item in series.index if 'factor' in item]
            factors=series[factor].sort_values()
            buy = [item.split('_')[0] for item in factors.iloc[:2].index]
            sell =[item.split('_')[0] for item in factors.iloc[-2:].index]
            money = moneys/(len(buy)+len(sell))
            for each in buy:
                stock[each]=money//series[each+'_open']
                remain=money-stock[each]*series[each+'_open']*(1+fee)
                remains+=remain
            for each in sell:
                stock[each]=-(money//series[each+'_open'])
                remain=money-stock[each]*series[each+'_open']*(1-fee)
                remains+=remain
            stocktotal=0
            for key,value in stock.items():
                stocktotal += value*series[key+'_close']
            moneys = remains+stocktotal
            flag=1
            return moneys
        if flag==1:
            moneys=remains+sum([value*series[key+'_open'] for key, value in stock.items()])
            stock={}
            remains=0
            factor = [item for item in series.index if 'factor' in item]
            factors=series[factor].sort_values()
            buy = [item.split('_')[0] for item in factors.iloc[:2].index]
            sell =[item.split('_')[0] for item in factors.iloc[-2:].index]
            money = moneys/(len(buy)+len(sell))
            for each in buy:
                stock[each]=money//series[each+'_open']
                remain=money-stock[each]*series[each+'_open']*(1+fee)
                remains+=remain
            for each in sell:
                stock[each]=-(money//series[each+'_open'])
                remain=money-stock[each]*series[each+'_open']*(1-fee)
                remains+=remain
            stocktotal=0
            for key,value in stock.items():
                stocktotal += value*series[key+'_close']
            moneys = remains+stocktotal
            flag=1
            return moneys    
    else:
        if flag==1:
            moneys=remains+sum([value*series[key+'_close'] for key, value in stock.items()])
            return moneys
        else:
            print(moneys)
            return moneys






        
    return  pd.concat(items,axis=1,join='inner')
