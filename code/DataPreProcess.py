import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


def split_cat_num(data, cat=15):
    '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其
    归为类别特征中。'''
    categorical = data.select_dtypes(include='object')
    numerical = data.select_dtypes(exclude='object')
    nunique = numerical.nunique().sort_values(ascending=True)
    n_index = nunique[nunique<cat].index
    num = numerical.columns.difference(n_index)
    category = categorical.columns
    return category, num, n_index


import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
class FeatureStatistics:
    '''从两个Series中生存DataFrame，另一种方法，不需要提前建立空DataFrame 
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100) 
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])'''
    def __init__(self):
        # 初始化存储类别和数值特征的DF
        self.num = pd.DataFrame(columns=['数据名','空值比例','偏度','峰度',
                        '类别数','高频类别','高频类别比例'])
        self.cat = pd.DataFrame(columns=['数据名','空值比例','类别数','高频类别',
                        '高频类别比例','熵'])
        
    def split(self,data): #划分类被和数值特征
        numerical = data.select_dtypes(exclude='object')
        categorical = data.select_dtypes(include='object')
        return numerical, categorical
    
    def describe(self, data): # 得到数值和类别特征的一些统计特征
        numerical,categorical = self.split(data)
        length = numerical.shape[0]
        if not numerical.empty:
            self.num['数据名'] = numerical.columns
            self.num = self.num.set_index('数据名')
            self.num['空值比例'] = pd.isnull(numerical).sum()/length
            self.num['类别数'] = numerical.nunique()
            self.num['偏度'] = numerical.skew()
            self.num['峰度'] = numerical.kurt()
            self.num['高频类别']= numerical.apply(lambda x: x.value_counts(
            ).sort_values(ascending=False).index[0])
            self.num['高频类别比例'] = (numerical==self.num['高频类别']).sum(
               )/length
            self.num = self.num.reset_index()
        if not categorical.empty:
            self.cat['数据名'] = categorical.columns
            self.cat = self.cat.set_index('数据名')
            self.cat['空值比例'] = pd.isnull(categorical).sum()/length
            self.cat['类别数'] = categorical.nunique()
            self.cat['高频类别']= categorical.apply(lambda x: x.value_counts(
                ).sort_values(ascending=False).index[0])
            self.cat['高频类别比例'] = (categorical==self.cat['高频类别']).sum(
                )/length
            self.cat['熵'] = categorical.apply(lambda x:stats.entropy(
            x.value_counts(normalize=True), base=2))
            self.cat = self.cat.reset_index()
            
    def plot(self, data):# 可视化类别和数值特征，数值默认为分布图，类别默认为柱状图
        numerical,categorical = self.split(data)
        g = None
        if not numerical.empty:
            melt = pd.melt(numerical)
            g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, 
                          sharex=False, sharey=False)
            g.map(sns.distplot,'value')
        if not categorical.empty:
            melt = pd.melt(categorical)
            g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, 
                          sharex=False, sharey=False)
            g.map(self.catplot,'value')
            
    def catplot(self,x,**kwarg):
        count = x.value_counts()
        ax = plt.gca()
        if count.shape[0]>50:
            sns.barplot(x=count.index,y=count.values,ax=ax,**kwarg)
            ax.xaxis.set_ticklabels([])
        else:
            sns.barplot(x=count.index,y=count.values,ax=ax,**kwarg)


class Constant:

    def __init__(self, deletes=0.9):
        self.deletes = deletes

    def check_constant(self, data):
        '''检测常变量，返回特征值为常变量或者缺失率为100%的特征。'''
        nuniq = data.nunique(dropna=False)
        drop_columns = nuniq[nuniq==1].index
        return drop_columns
    
    def most_frequent(self, data):
        '''计算每个特征中出现频率最高的项所占的比例和对应的值'''
        records_count = data.shape[0]
        col_most_values, col_large_value = {},{}
        
        for col in data.columns:
            value_counts = data[col].value_counts()
            col_most_values[col] = value_counts.max()/records_count
            col_large_value[col] = value_counts.idxmax()

        most_values_df = pd.DataFrame.from_dict(col_most_values, 
                                        orient = 'index')
        most_values_df.columns = ['max percent']
        most_values_df = most_values_df.sort_values(
            by = 'max percent', ascending = False)
        return most_values_df, col_large_value

    def plot_bin(self, data, ax=None):
        data.columns=['a','b']
        ax.vlines(x=data.index, ymin=0, ymax=data['b'], 
                      color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'], s=75, 
                       color='firebrick', alpha=0.7)
        ax.set_title('Missing Rate', 
                         fontdict={'size':22})
        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'], 
                    rotation=90, 
                fontdict={'horizontalalignment': 'right', 'size':12})
        for row in data.itertuples():
            ax.text(row.Index, row.b*1.01, s=round(row.b, 1),
                    horizontalalignment= 'center',
                    verticalalignment='bottom', fontsize=14)

    def plot_frequency(self, data, N=30):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 可以显示中文
        plt.rcParams['axes.unicode_minus']=False # 可以显示负号
        _, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.set_ylabel('Frequency Rate')
        ser, _ = self.most_frequent(data)
        ser=ser[:N]
        data=ser.reset_index()
        self.plot_bin(data, ax=ax)

    def frequence_bin(self, data, features):
        '''对特征进行0-1编码的特征，出现次数最多的的样本为一类，其他的为一类'''
        result = data[features].copy()
        col_large = { }
        for each in features:
            value_counts = result[each].value_counts()
            col_large[each] = value_counts.idxmax()
            result[each+'_bins'] = (result[each]== col_large[each]).astype(int)
        return result, col_large

    def delete_frequency(self, data):
        '''对于没有明显差异，同时特征中某个值出现的概率很高的特征进行删除。 '''
        result = data.copy()
        col_most,_ = self.most_frequent(result)
        large_percent_cols = list(col_most[col_most['max percent']>=self.deletes].index)
        result = result.drop(large_percent_cols, axis=1)
        return result, large_percent_cols

    def fit(self, X, y=None):
        col_most, col_large_value = self.most_frequent(X)
        large_percent_cols = list(col_most[col_most['max percent']>=self.deletes].index)
        self.large_percent_ = large_percent_cols
        self.col_large_ = col_large_value
        return self
    
    def transform(self, X):
        result = X.copy()
        result = result.drop(self.large_percent_, axis=1)
        return result
    
    def fit_transform(self, X, y=None):
        result = X.copy()
        col_most, col_large_value = self.most_frequent(X)
        large_percent_cols = list(col_most[col_most['max percent']>=self.deletes].index)
        self.large_percent_ = large_percent_cols
        self.col_large_ = col_large_value
        result = result.drop(large_percent_cols, axis=1)
        return result



class Missing:
    
    def __init__(self, delete=0.9, indicator=0.6, fill=0.1):
        '''初始化三个阈值,删除的阈值，产生indicator的阈值，填充的阈值。'''
        self.delete = delete #特征删除的阈值，如果缺失率大于这个值，则删除
        self.indicator = indicator # 缺失值进行编码的阈值。
        self.fill = fill # 缺失值填充的阈值
        self.delete_ = None # 记录删除特征，用于测试集进行数据处理
        self.indicator_ = None # 记录编码的特征
        self.fill_value_={} # 记录填充的值
        
    def is_null(self, data):
        '''检验数据集中每一个元素是否为空值'''
        null = data.isnull()
        null_sum = null.sum()
        null_item = null.sum(axis=1)
        return null, null_sum, null_item
    
    def plot_miss(self, data):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 可以显示中文
        plt.rcParams['axes.unicode_minus']=False # 可以显示负号
        _, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.set_ylabel('Rate')
        ser = (data.isnull().sum()/data.shape[0]).sort_values(
            ascending=False)
        ser=ser[ser>0]
        data=ser.reset_index()
        data.columns=['a','b']
        ax.vlines(x=data.index, ymin=0, ymax=data['b'], 
                      color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'], s=75, 
                       color='firebrick', alpha=0.7)
        ax.set_title('Missing Rate', 
                         fontdict={'size':22})
        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'], 
                    rotation=90, 
                fontdict={'horizontalalignment': 'right', 'size':12})
        for row in data.itertuples():
            ax.text(row.Index, row.b*1.01, s=round(row.b, 2),
                    horizontalalignment= 'center',
                    verticalalignment='bottom', fontsize=14)
    
    def plot_item_miss(self,data):
        '''将每个样本的缺失值个数按从小到大绘制出来。'''
        ser = (data.isnull().sum(axis=1)).sort_values()
        x = range(data.shape[0])
        plt.scatter(x,ser.values,c='black')
        
    def find_index(self, data, threshold=None):
        '''找到满足条件的特征'''
        length = data.shape[0]
        _, null_sum, _ = self.is_null(data)
        ratio = null_sum()/length
        index = ratio[ratio>=threshold].index
        return index
    
    def delete_null(self, data, threshold=None):
        '''删除缺失比例较高的特征，同时将缺失比例较高的样本作为缺失值删除。'''
        result = data.copy()
        if threshold is None:
            threshold = self.delete
        index = self.find_index(data, threshold)
        result = result.drop(index, axis=1)
        return index, result
    
    def delete_items(self, data, value=None):
        '''删除包含缺失值较多的样本，value为删除的阈值，如果样本的缺失值个数大于value
        则将其视为异常值删除。'''
        data = data.copy()
        *_, null_item = self.is_null(data)
        index2 = null_item[null_item>value].index
        data = data.drop(index2)
        return data
    
    def indicator_null(self, data, threshold=None):
        '''生产特征是否为缺失的指示特征，同时删除原特征。'''
        result = data.copy( )
        if threshold is None:
            threshold = self.indicator
        index = self.find_index(data, threshold)
        for each in index:
            result['is_null_'+each] = pd.isnull(result[each]).astype(int)
        result = result.drop(index, axis=1)
        return index, result
    
    def another_class(self, data, features=None):
        '''对于类别特征而言，所有缺失值另作一类'''
        result = data.loc[:,features].copy()
        fill_value = { }
        for each in features:
            if data[each].dtype == 'object':
                fill_value[each] = 'None'
            else:
                fill_value[each] = int(data[each].max()+1)
        result.fillna(self.fill_value_)
        return fill_value, result

    def bin_and_fill(self, data, features):
        '''对于一些数值特征，我们可以使用中位数填补，但是为了不丢失缺失信息，同时可以进行编码。'''
        result = data.loc[:,features].copy()
        fill_value = { }
        for each in features:
            result['is_null_'+each] = pd.isnull(data[each]).astype(int)
            fill_value[each] = data[each].median()
            result[each] = data[each].fillna(fill_value[each])
        return fill_value, result
    
    def fill_null(self, data, features):
        '''对于缺失率很小的数值特征，使用中位数填补缺失值'''
        result = data.loc[:, features].copy()
        fill_value = result.median().to_dict()
        result = result.fillna(result.median())
        return fill_value, result
    
    def fit(self, X, y=None):
        data = X.copy()
        index, result = self.delete_null(data)
        self.delete_ = index
        index2, _ = self.indicator_null(result)
        self.indicator_ = index2
        return self
        
    def transform(self, X):
        data = X.copy()
        data = data.drop(self.delete_, axis=1)
        data = data.drop(self.indicator_, axis=1)
        for each in self.indicator_:
            data['is_'+each] = pd.isnull(data[each]).astype(int)
        return data
    
    def fit_transform(self, X, y=None):
        data = X.copy()
        index, result = self.delete_null(data)
        self.delete_ = index
        index2, result2 = self.indicator_null(result)
        self.indicator_ = index2
        return result2


from matplotlib import gridspec    
    
class Outlier:
    def __init__(self, method=sns.stripplot):
        self.method = method
    
    def plot_num_data(self, data, target, features=None):
        '''按照label,对数据集中的数值特征绘制stripplot，可以根据图形从中寻找到
        数值特征中的异常值。'''
        if features is None:
            _,num, _ = split_cat_num(data)
        else:
            num = features
        data = pd.concat([data, target], axis=1)
        name = target.name
        melt = pd.melt(data, id_vars=name, 
                       value_vars = [f for f in num])
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, 
                          sharex=False, sharey=False)
        g.map(self.method, name, 'value', jitter=True,
              palette="muted",order=[0,1])
              
    def plot_cat_data(self, data, y, features=None):
        if features is None:
            cat, _, n_index = split_cat_num(data)
            cols = cat.union(n_index)
        else: cols = features
        self.ploting_cat_fet(data, cols, y)

    def ploting_cat_fet(self, df, cols, y):
        '''绘制类别特征，柱状图为每个值在特征中出现的数目及所占的比例，折线图为每个取值
        情况下，其中的坏样本率。'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 可以显示中文
        plt.rcParams['axes.unicode_minus']=False # 可以显示负号
        total = len(df)
        # 图形的参数设置
        nrows, ncols = len(cols)//2+1, 2
        grid = gridspec.GridSpec(nrows,ncols) 
        fig = plt.figure(figsize=(16, 20*nrows//3))
        df = pd.concat([df,y],axis=1)
        name = y.name
        
        for n, col in enumerate(cols): 
            tmp = pd.crosstab(df[col], df[name], 
                            normalize='index') * 100
            tmp = tmp.reset_index()
            tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

            ax = fig.add_subplot(grid[n])
            sns.countplot(x=col, data=df, order=list(tmp[col].values) , 
                        color='green') # 绘制柱状图
            
            ax.set_ylabel('Count', fontsize=12) # 设置柱状图的参数
            ax.set_title(f'{col} Distribution by Target', fontsize=14)
            ax.set_xlabel(f'{col} values', fontsize=12)
            
            gt = ax.twinx() # 绘制折线图
            gt = sns.pointplot(x=col, y='Yes', data=tmp,
                            order=list(tmp[col].values),
                            color='black', legend=False)
            
            mn,mx = gt.get_xbound()# 绘制水平线
            gt.hlines(y=y.sum()/total,xmin=mn,xmax=mx,color='r', linestyles='--')
            
            gt.set_ylim(0,tmp['Yes'].max()*1.1) # 设置y轴和title
            gt.set_ylabel("Target %True(1)", fontsize=16)
            sizes=[] 
            
            for p in ax.patches: # 标识每个值所占的比例。
                height = p.get_height()
                sizes.append(height)
                ax.text(p.get_x()+p.get_width()/2.,
                        height + 3,
                        '{:1.2f}%'.format(height/total*100),
                        ha="center", fontsize=12) 
            ax.set_ylim(0, max(sizes) * 1.15) 
        plt.subplots_adjust(hspace = 0.5, wspace=.3)