import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec


class ConstantBase:

    def check_constant(self, data):
        '''检测常变量，返回特征值为常变量或者缺失率为100%的特征。'''
        nuniq = data.nunique(dropna=False)
        drop_columns = nuniq[nuniq == 1].index
        return drop_columns

    def most_frequent(self, data):
        '''计算每个特征中出现频率最高的项所占的比例和对应的值'''
        col_most_values, col_large_value = {}, {}

        for col in data.columns:
            value_counts = data[col].value_counts(normalize=True)
            col_most_values[col] = value_counts.max()
            col_large_value[col] = value_counts.idxmax()

        most_values_df = pd.DataFrame.from_dict(col_most_values,
                                                orient='index')  # 字典的key为index
        most_values_df.columns = ['max percent']
        most_values_df = most_values_df.sort_values(by='max percent',
                                                    ascending=False)
        return most_values_df, col_large_value
    
    def find_columns(self, data, threshold=None):
        if threshold is None:
            threshold = self.deletes
        col_most, _ = self.most_frequent(data)
        large_percent_cols = list(
            col_most[col_most['max percent'] >= threshold].index)
        return large_percent_cols


class ConstantProcess(ConstantBase):

    def __init__(self, deletes=0.9):
        self.deletes = deletes


    def plot_frequency(self, data, N=30):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.set_ylabel('Frequency Rate')
        ser, _ = self.most_frequent(data)
        ser = ser[:N]
        data = ser.reset_index()
        ax = self.plot_bin(data, ax=ax)
        ax.set_title('Most Frequency', fontdict={'size': 18})

    def frequence_bin(self, data, features, col_large=None):
        '''对特征进行0-1编码的特征，出现次数最多的的样本为一类，其他的为一类'''
        result = data[features].copy()
        if col_large is None:
            col_larges = {}
            for each in features:
                col_large = result[each].value_counts().idxmax()
                col_larges[each] = col_large
                result[each+'_bins'] = (result[each] == col_large).astype(int)
            return result, col_larges
        else:
            for each in features:
                result[each +
                       '_bins'] = (result[each] == col_large[each]).astype(int)
            return result

    def delete_frequency(self, data, threshold=None):
        '''特征中某个值出现的概率很高的特征进行删除。 '''
        large_percent_cols = self.find_columns(data, threshold)
        result = data.copy()
        result = result.drop(large_percent_cols, axis=1)
        return result, large_percent_cols

    def fit(self, X, y=None, threshold=None):
        large_percent_cols = self.find_columns(X, threshold)
        self.large_percent_ = large_percent_cols
        return self

    def transform(self, X):
        result = X.copy()
        assert hasattr(self, 'large_percent_')
        result = result.drop(self.large_percent_, axis=1)
        return result

    def fit_transform(self, X, y=None, threshold=None):
        result = X.copy()
        large_percent_cols = self.find_columns(X, threshold)

        self.large_percent_ = large_percent_cols
        result = result.drop(large_percent_cols, axis=1)
        return result