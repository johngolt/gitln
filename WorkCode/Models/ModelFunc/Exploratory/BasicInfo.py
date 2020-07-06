import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec


class FeatureStatistics:
    '''得到数据的基本信息以及对可视化数据的分布。'''

    def split(self, data): 
        '''将object类型和数值类型分开'''
        numerical = data.select_dtypes(exclude='object')
        categorical = data.select_dtypes(include='object')
        return numerical, categorical

    def describe(self, data):
        '''得到数值和类别特征的一些统计特征'''
        numerical, categorical = self.split(data)
        if not numerical.empty:
            _ = self.describe_num(numerical)
        if not categorical.empty:
            _ = self.describe_cat(categorical)

    def _describe_all(self, data):
        '''类别特征和数值特征共有的一些特征信息，包括数据名，空值比例，取值个数，高频比例
        高频数，同时记录样本数数。'''
        length = data.shape[0]
        result = pd.DataFrame(columns=['数据名', '空值比例', '类别数', '高频类别', '高频类别比例'])
        result['数据名'] = data.columns
        result = result.set_index('数据名')
        result['空值比例'] = pd.isnull(data).sum()/length
        result['类别数'] = data.nunique()
        result['高频类别'] = data.apply(lambda x: x.value_counts().idxmax())
        result['高频类别比例'] = (data == result['高频类别']).sum()/length
        return result, length

    def describe_num(self, numerical, stat=False):
        '''得到数值特征的统计特征，除去共有的特征信息外，增加了负值比例，零值比例，
        也可以根据具体情况也可以查看数值特征的一些统计特征：最大最小值，均值等。'''
        self.num, length = self._describe_all(numerical)
        self.num['负值比例'] = (numerical < 0).sum()/length
        self.num['零值比例'] = (numerical == 0).sum()/length
        if stat:
            statis = self.statistic(numerical)
            self.num = self.num.join(statis)
        self.num = self.num.reset_index()
        return self.num

    def statistic(self, numerical):
        '''得到数值特征的一些统计特征，为了展示的简洁，这部分信息可以选择性的查看。'''
        stat = pd.DataFrame(
            columns=['数据名', '最大值', '最小值', '中位数', '均值', '偏度', '峰度'])
        stat['数据名'] = numerical.columns
        stat = stat.set_index('数据名')
        stat['最小值'] = numerical.min()
        stat['最大值'] = numerical.max()
        stat['均值'] = numerical.mean()
        stat['中位数'] = numerical.median()
        stat['偏度'] = numerical.skew()
        stat['峰度'] = numerical.kurt()
        return stat

    def describe_cat(self, categorical):
        '''得到类别特征的信息，除去共有的特征信息外，增加了熵值，
        越接近0代表分布越集中，越接近1代表分布越分散。'''
        self.cat, _ = self._describe_all(categorical)
        self.cat['熵'] = categorical.apply(lambda x: stats.entropy(
            x.value_counts(normalize=True), base=2))/np.log2(self.cat['类别数'])
        self.cat = self.cat.reset_index()
        return self.cat

    def plot(self, data):
        '''可视化类别和数值特征，数值默认为分布图，类别默认为柱状图'''
        numerical, categorical = self.split(data)
        if not numerical.empty:
            self.plot_numerical(numerical)
        if not categorical.empty:
            self.plot_categories(categorical)

    def plot_numerical(self, numerical, style=sns.distplot, **kwargs):
        '''数值特征的可视化，包括kdeplot，默认不可以有缺失值。'''
        melt = pd.melt(numerical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(style, 'value', **kwargs)

    def plot_categories(self, categorical, **kwargs):
        '''类别特征的可视化，对于取值较多的类别特征，这里面定义为大于30个，为了
        可视化的效果，将绘制出现频率的散点图，小于等于30个将绘制条形图。可视化函数
        由_catplot函数实现。'''
        melt = pd.melt(categorical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(self._catplot, 'value', **kwargs)

    def _catplot(self, ser, **kwarg):  # 类别特征的可视化
        count = ser.value_counts()
        ax = plt.gca()
        if count.shape[0] > 30:  # 对于取值较多的特征，绘制散点图。
            ax.scatter(x=count.index, y=count.values, **kwarg)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.xaxis.set_ticklabels([])
        else:  # 其他情况下绘制条形图。
            sns.barplot(x=count.index, y=count.values, ax=ax, **kwarg)



