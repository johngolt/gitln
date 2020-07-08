import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号


class MissingBase:
    '''对特征中缺失值的处理方法，包括缺失值的可视化，特征缺失的可视化和样本缺失的可视化，对于缺失值的处理分为：
    删除，0-1编码，另作为一类，编码并填补，填补缺失值。'''

    def is_null(self, data):
        return data.isnull()
    
    def is_null_feature(self, data):
        null = self.is_null(data)
        return null.sum()
    
    def is_null_item(self, data):
        return self.is_null(data).sum(axis=1)
    
    def report(self, data):
        ratio = self.is_null_feature(data)/data.shape[0]
        ratio = ratio.sort_values(ascending=False).reset_index()
        ratio.columns = ['特征', '比例']
        ratio['比例'] = ratio['比例'].map(lambda x: '{:.2%}'.format(x))
        return ratio


class MissingPlot(MissingBase):

    def plot_miss(self, data):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        _, ax = plt.subplots(figsize=(8, 5))
        ax.set_ylabel('Missing Rate')
        null = self.is_null_feature(data)
        n = data.shape[0]
        ser = (null/n).sort_values(ascending=False)
        ser = ser[ser > 0]
        data = ser.reset_index()
        ax = self.plot_bin(data, ax=ax)
        ax.set_title('Missing Rate', fontdict={'size': 18})

    def plot_item_miss(self, data):
        '''将每个样本的缺失值个数按从小到大绘制出来。'''
        null = self.is_null_item(data)
        ser = null.sort_values()
        x = range(data.shape[0])
        plt.scatter(x, ser.values, c='black')


class MissingProcess(MissingBase):

    def __init__(self, delete=0.9, indicator=0.6, fill=0.1):
        '''初始化三个阈值,删除的阈值，产生indicator的阈值，填充的阈值。'''
        self.delete = delete  # 特征删除的阈值，如果缺失率大于这个值，则删除
        self.indicator = indicator  # 缺失值进行编码的阈值。
        self.fill = fill  # 缺失值填充的阈值
        self.delete_ = None  # 记录删除特征，用于测试集进行数据处理
        self.indicator_ = None  # 记录编码的特征
        self.fill_value_ = {}  # 记录填充的值


    def find_index(self, data, threshold):
        '''找到满足条件的特征'''
        length = data.shape[0]
        null_sum = self.is_null_feature(data)
        ratio = null_sum/length
        index = ratio[ratio >= threshold].index
        return index

    def delete_null(self, data, threshold=None, index=None):
        '''删除缺失比例较高的特征，同时将缺失比例较高的样本作为缺失值删除。'''
        result = data.copy()
        if threshold is None:
            threshold = self.delete
        if index is None:
            index = self.find_index(data, threshold)
        result = result.drop(index, axis=1)
        return index, result

    def delete_items(self, data, value):
        '''删除包含缺失值较多的样本，value为删除的阈值，如果样本的缺失值个数大于value
        则将其视为异常值删除。'''
        result = data.copy()
        null_item = self.is_null_item(data)
        index2 = null_item[null_item > value].index
        result = result.drop(index2)
        return result

    def indicator_null(self, data, threshold=None, index=None):
        '''生产特征是否为缺失的指示特征，同时删除原特征。'''
        result = data.copy()
        if threshold is None:
            threshold = self.indicator
        if index is None:
            index = self.find_index(data, threshold)
        for each in index:
            result['is_null_'+each] = pd.isnull(result[each]).astype(int)
        result = result.drop(index, axis=1)
        return index, result

    def another_class(self, data, features, fill_value=None):
        '''对于特征而言，可以将缺失值另作一类'''
        result = data.loc[:, features].copy()
        if fill_value is None:
            fill_value = {}
            for each in features:
                if data[each].dtype == 'object':
                    fill_value[each] = 'None'
                else:
                    fill_value[each] = int(data[each].max()+1)
        result = result.fillna(fill_value)
        return fill_value, result

    def bin_and_fill(self, data, features, fill_value=None):
        '''对于一些数值特征，我们可以使用中位数填补，但是为了不丢失缺失信息，同时可以进行编码。'''
        result = data.loc[:, features].copy()
        if fill_value is None:
            fill_value = result[features].median().to_dict()
        for each in features:
            result['is_null_'+each] = pd.isnull(data[each]).astype(int)
            result[each] = data[each].fillna(fill_value[each])
        return fill_value, result

    def fill_null(self, data, features, fill_value=None):
        '''对于缺失率很小的数值特征，使用中位数填补缺失值,可以自定义填充，
        默认采用中位数填充。'''
        result = data.loc[:, features].copy()
        if fill_value is None:
            fill_value = result.median().to_dict()
        result = result.fillna(fill_value)
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