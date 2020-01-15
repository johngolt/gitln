'''
这一模块主要包括一些数据处理的基本方法，用于建模过程中的数据分析和处理。主要包括以下几部分。
split_cat_num:将特征进行分类，分为数值特征，类别特征和数值特征中取值较少可能为类别特征的特征。 

FeatureStatistics:主要对特征进行一些简单的信息统计和可视化，按照类别和数值特征两种来进行处理。

PlotFunc:包含了一些绘图的基本处理，方便后续数据可视化。

Categorical:主要针对类别特征的可视化，用于数据预处理阶段。

Numerical:主要针对于数值特征的可视化，用于数据预处理阶段。

Constant:针对于常值特征的处理和可视化。

CalEnt:计算类别特征的条件信息熵，基尼系数和WOE,IV值

Missing:针对于数据处理过程中的缺失值处理，包含了常用的一些缺失值处理方法。

FeatureStability:用于检验训练集和测试集中特征的稳定性，可以作为特征选择的一部分，也可以用于查看
特征在不同类别之间的分布情况。
'''

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


def split_cat_num(data, cat=15):
    '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其归为类别特征中。'''
    categorical = data.select_dtypes(include='object')
    numerical = data.select_dtypes(exclude='object')
    nunique = numerical.nunique().sort_values(ascending=True)
    n_index = nunique[nunique < cat].index
    num = numerical.columns.difference(n_index)
    category = categorical.columns
    return category, num, n_index


class FeatureStatistics:
    '''得到数据的基本信息以及对可视化数据的分布。'''

    def split(self, data):  # 划分类别和数值特征
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
        '''数值特征的可视化，包括kdeplot'''
        melt = pd.melt(numerical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(style, 'value', **kwargs)

    def plot_categories(self, categorical, **kwargs):
        '''类别特征的可视化，对于取值较多的类别特征，这里面定义为大于50个，为了
        可视化的效果，将绘制出现频率的散点图，小于等于50个将绘制条形图。可视化函数
        由_catplot函数实现。'''
        melt = pd.melt(categorical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(self._catplot, 'value', **kwargs)

    def _catplot(self, ser, **kwarg):  # 类别特征的可视化
        count = ser.value_counts()
        ax = plt.gca()
        if count.shape[0] > 50:  # 对于取值较多的特征，绘制散点图。
            ax.scatter(x=count.index, y=count.values, **kwarg)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.xaxis.set_ticklabels([])
        else:  # 其他情况下绘制条形图。
            sns.barplot(x=count.index, y=count.values, ax=ax, **kwarg)


class PlotFunc:
    '''针对类别特征和数值特征可视化的类别，更加具体的可视化特征的一些信息。'''
    def __init__(self):
        self.font1 = {'family': 'Calibri', 'weight': 'normal', 'size': 14}
        self.font2 = {'family': 'Calibri', 'weight': 'normal', 'size': 18}

    def get_ax(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot()
            return ax
        return ax

    def plot_bin(self, data, ax=None):
        ax = self.get_ax(ax)
        data.columns = ['a', 'b']
        ax.vlines(x=data.index, ymin=0,
                  ymax=data['b'], color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'],
                   s=75, color='firebrick', alpha=0.7)

        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'], rotation=90,
                        fontdict={'horizontalalignment': 'right', 'size': 12})

        for row in data.itertuples():
            ax.text(row.Index, row.b*1.01, s=round(row.b, 2), 
            horizontalalignment='center', verticalalignment='bottom', fontsize=14)
        return ax


class Categorical(PlotFunc):
    '''对类别特征进行分析，适合取值不多的类别特征。对于取值很多的类别特征可视化效果不好'''

    def _crosstab(self, data, feature, target, ax=None):
        '''查看目标在类别特征的每个值的分布情况。'''
        ax = self.get_ax(ax)
        ct = pd.crosstab(data[feature], data[target], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('{}'.format(feature))
        ax.axhline(1-data[target].mean(), color="crimson", alpha=0.9, linestyle="--")
        ax.set_ylim(0, 1)
        plt.ylabel('{} Rate'.format(target))
        ax.set_xlabel('')
        return ax

    def plot_crosstab(self, data, features, target):
        nrows = len(features)//4+1
        fig = plt.figure(figsize=(20, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=4)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            self._crosstab(data, each, target, ax=ax)
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

    def plot_dists(self, data, features, target):
        nrows = len(features)
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_dist(data, each, target, [ax, ax1])
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_dist(self, data, feature, target, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.subplots(1, 2)
        _ = self.plot_bar(data, feature, ax[0])
        _ = self.plot_bar_line(data, feature, target, ax[1])

    def plot_bar(self, data, feature, ax1=None):
        '''用条形图展示值分布。在类别特征中，若某个类别所含样品频数特别少，则可以认为是异常值，舍弃。
        若有多个类别的样本频数都较少，可以考虑合并类别。'''
        ax1 = self.get_ax(ax1)
        df = data[feature].value_counts().sort_values()
        df.plot(kind='bar', ax=ax1)
        ax1.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax1.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax1.set_ylim(bottom=0, top=len(data))
        ax1.legend()
        xs = ax1.xaxis.get_ticklocs()
        for x, y in zip(xs, df.values):
            ax1.annotate(s=str(y), xy=(x, y), xytext=(x*0.95, y*1.01))
        return ax1

    def plot_bar_line(self, data, feature, target, ax=None):
        '''可视化类别特征中样本的频率，和目标变量的分布情况。'''
        ax = self.get_ax(ax)
        df = pd.crosstab(data[feature], data[target])
        ax = df.plot(kind='bar', stacked=True, ax=ax, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax.set_ylim(bottom=0, top=len(data))
        ax.legend(loc=2)

        ax.xaxis.set_ticklabels(df.index)
        xs = ax.xaxis.get_ticklocs()
        odds = df[1]/df[0]
        axt = ax.twinx()
        axt.plot(xs, odds, marker="o")
        axt.set_ylabel("odds", fontdict=self.font1, labelpad=6)
        odd = df.sum()[1]/df.sum()[0]
        axt.axhline(odd, color="crimson", alpha=0.9, linestyle="--")
        return ax


class Numerical(PlotFunc):

    def drop_null_item(self, data, feature=None):
        '''丢弃特征中确实的样本。'''
        if feature is None:
            return data[data.notnull()]
        temp = data.loc[data[feature].notnull(), feature]
        return temp

    def _kstest(self, data, feature): 
        '''数值特征的正态性检验，确定分布是否符合正态分布。'''
        mean, std = data[feature].mean(), data[feature].std()
        temp = self.drop_null_item(data, feature)
        _, pvalue = stats.kstest(temp, stats.norm(mean, std).cdf)
        if pvalue < 0.05:
            return False
        else:
            return True

    def kstests(self, data, features):
        mask = [self._kstest(data, each) for each in features]
        return mask
    
    def plot_strips(self, data, features, target, ax=None):
        '''按照label,对数据集中的数值特征绘制stripplot，可以根据图形从中寻找到
        数值特征中的异常值。'''
        nrows, ncols = len(features)//2+1, 2
        grid = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(16, 20*nrows//3))
        for i, feature in enumerate(features):
            ax = fig.add_subplot(grid[i])
            sns.stripplot(target, feature, jitter=True, palette='muted',
                       order=[0,1], data=data, ax=ax)

    def plot_kde(self, data, feature, ax=None, **kwargs):
        '''绘制数值特征的kernel density estimation，同时采用
        正太分布进行对照。'''
        ax = self.get_ax(ax)
        sample = self.drop_null_item(data, feature).to_numpy()
        ax = sns.distplot(sample,hist=False, fit=stats.norm, ax=ax, **kwargs)
        ax.legend(ax.lines, ['kde','norm'],loc=1)
        ax.set_title('{} kde distribution'.format(
            feature), fontdict=self.font1)
        ax.set_ylabel('Probability', fontdict=self.font2, labelpad=6)
        return ax

    def plot_kdes(self, data, features, **kwargs):
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            self.plot_kde(data, each, ax=ax, **kwargs)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_boxs(self, data, features, **kwargs):
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            _ = self.plot_box(data, each, ax=ax, **kwargs)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_box(self, data, feature, ax=None, **kwargs):
        '''绘制数值特征的箱型图。'''
        ax = self.get_ax(ax)
        sns.boxplot(data[feature], ax=ax, orient='v', **kwargs)
        ax.set_title('{} boxplot'.format(feature), fontdict=self.font2)
        ax.set_ylabel('{}'.format(feature), fontdict=self.font1, labelpad=6)
        return ax

    def plot_hists(self, data, features, target, bins=50):
        nrows = len(features)
        fig = plt.figure(figsize=(8, 4*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            self.plot_hist(data, each, ax=ax, bins=bins)
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_hist_line(data, each, target, ax=ax1, bins=bins)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_hist(self, data, feature, ax=None, bins=50):
        '''绘制数值特征的条形图'''
        ax = self.get_ax(ax)
        temp = self.drop_null_item(data, feature)
        bins = min(int(np.sqrt(temp.shape[0])),bins)
        _ = ax.hist(data[feature], bins=bins, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_xlabel('{}'.format(feature), fontdict=self.font1, labelpad=2)
        ax.set_ylabel('frequency', fontdict=self.font1, labelpad=6)
        return ax

    def plot_hist_line(self, data, feature, target, ax=None, bins=50):
        '''绘制在目标变量下数值特征的条形图，同时查看目标在特征不同区间下的分布。'''
        ax = self.get_ax(ax)
        X0 = data.loc[data[target] == 0, feature]
        X1 = data.loc[data[target] == 1, feature]
        bins = min(int(np.sqrt(min(len(X0), len(X1)))),bins)
        X0 = self.drop_null_item(X0)
        X1 = self.drop_null_item(X1)
        n1, bins1, _ = ax.hist(X0, bins=bins, alpha=0.6, label='Target=0')
        n2, *_ = ax.hist(X1, bins=bins1, alpha=0.6,
                         bottom=n1, label='Target=1')
        ax.set_title('{} distribution in {}'.format(
            target, feature), fontdict=self.font2)
        ax.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax.legend(loc=2)
        odds = n2.sum()/n1.sum()
        xs, ys = (bins1[:-1]+bins1[1:])/2, (n2+odds)/(n1+1)
        axt = ax.twinx()
        axt.plot(xs, ys, marker='*')
        axt.axhline(odds, color='crimson', alpha=0.8, linestyle='--')
        axt.set_ylabel('Odds', fontdict=self.font1, labelpad=6)
        return ax


class Constant(PlotFunc):

    def __init__(self, deletes=0.9):
        self.deletes = deletes

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

        most_values_df = pd.DataFrame.from_dict(col_most_values, orient='index')  # 字典的key为index
        most_values_df.columns = ['max percent']
        most_values_df = most_values_df.sort_values(by='max percent', ascending=False)
        return most_values_df, col_large_value

    def plot_frequency(self, data, N=30):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        _, ax = plt.subplots(figsize=(8, 6))
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
            col_larges={}
            for each in features:
                col_large = result[each].value_counts().idxmax()
                col_larges[each] = col_large
                result[each+'_bins'] = (result[each] == col_large).astype(int)
            return result, col_larges
        else:
            for each in features:
                result[each+'_bins'] = (result[each] == col_large[each]).astype(int)
            return result
    
    def find_columns(self, data, threshold=None):
        if threshold is None:
            threshold = self.deletes
        col_most, _ = self.most_frequent(data)
        large_percent_cols = list(
            col_most[col_most['max percent'] >= threshold].index)
        return large_percent_cols

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


class CalEnt:
    '''比较某字段在类别合并前后的信息熵、基尼值、信息值IV，若合并后信息熵值减小/基尼值减小/信息值IV增大，
    则确认合并类别有助于提高此字段的分类能力，可以考虑合并类别。'''
    def _entropy(self, group): # 计算信息熵
        return -group*np.log2(group+1e-5)

    def _gini(self, group):  # 计算基尼系数
        return group*(1-group)

    def cal(self, data, feature, target, func=None):
        '''计算使用的通用函数'''
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = func(temp)
        return (temp1*enti.sum(axis=1)).sum()
    
    def calculates(self, data, features, target, func):
        '''通用函数，用来处理多个特征的计算'''
        if isinstance(features, str):
            return func(data, features, target)
        elif isinstance(features, Iterable):
            data = [func(data, feature, target) for feature in features]
            result = pd.Series(data, index=features)
            return result
        else:
            raise TypeError('Features is not right!')
    
    def entropy(self, data, feature, target):  # 计算条件信息熵
        return self.cal(data, feature, target, self._entropy)
    
    def entropys(self, data, features, target):
        '''计算条件信息熵的接口，通过条件信息熵可以评价特征的重要程度。'''
        return self.calculates(data, features, target, self.entropy)
    
    def gini(self, data, feature, target):  # 计算条件基尼系数
        return self.cal(data, feature, target, self._gini)
    
    def ginis(self, data, features, target):
        '''计算条件信息系数的接口，通过条件信息系数可以评价特征的重要程度。'''
        return self.calculates(data, features, target, self.entropy)
    
    def woe(self, data, feature, target):
        '''计算woe值,可以用于类别特征的编码'''
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = np.log((temp.iloc[:, 0]+1e-5)/(temp.iloc[:, 1]+1e-5))
        return woei

    def iv(self, data, feature, target):
        '''计算IV值，通过IV值可以进行特征选择，一般要求IV值大于0.02'''
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = self.woe(data, feature, target)
        iv = (temp.iloc[:, 0] - temp.iloc[:, 1])*woei
        return iv.sum()
    
    def ivs(self, data, features, target):
        return self.calculates(data, features, target, self.iv)
        
    def woes(self, data, features, target):
        if isinstance(features, str):
            return self.woe(data, features, target)
        elif isinstance(features, Iterable):
            return{each: self.woe(data, each, target) for each in features}
        else:
            raise TypeError('Feature is not right data type')


class Missing(PlotFunc):
    '''对特征中缺失值的处理方法，包括缺失值的可视化，特征缺失的可视化和样本缺失的可视化，对于缺失值的处理分为：
    删除，0-1编码，另作为一类，编码并填补，填补缺失值。'''
    def __init__(self, delete=0.9, indicator=0.6, fill=0.1):
        '''初始化三个阈值,删除的阈值，产生indicator的阈值，填充的阈值。'''
        self.delete = delete  # 特征删除的阈值，如果缺失率大于这个值，则删除
        self.indicator = indicator  # 缺失值进行编码的阈值。
        self.fill = fill  # 缺失值填充的阈值
        self.delete_ = None  # 记录删除特征，用于测试集进行数据处理
        self.indicator_ = None  # 记录编码的特征
        self.fill_value_ = {}  # 记录填充的值

    def is_null(self, data):
        '''检验数据集中每一个元素是否为空值,返回关于行列缺失情况的统计'''
        null = data.isnull()
        null_sum = null.sum()
        null_item = null.sum(axis=1)
        return null, null_sum, null_item

    def plot_miss(self, data):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号
        _, ax = plt.subplots(figsize=(8, 5))
        ax.set_ylabel('Missing Rate')
        ser = (data.isnull().sum()/data.shape[0]).sort_values(
            ascending=False)
        ser = ser[ser > 0]
        data = ser.reset_index()
        ax = self.plot_bin(data, ax=ax)
        ax.set_title('Missing Rate', fontdict={'size': 18})

    def plot_item_miss(self, data):
        '''将每个样本的缺失值个数按从小到大绘制出来。'''
        ser = (data.isnull().sum(axis=1)).sort_values()
        x = range(data.shape[0])
        plt.scatter(x, ser.values, c='black')

    def find_index(self, data, threshold):
        '''找到满足条件的特征'''
        length = data.shape[0]
        _, null_sum, _ = self.is_null(data)
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
        *_, null_item = self.is_null(data)
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


class FeatureStability:
    '''可视化特征在训练集和测试集上的分布，可以用来发现不稳定的特征。也可以用来可视化特征在不同类别的特征，
    用来选取重要特征或者删除不重要特征。通过函数发现训练和测试集分布不一致的特征，返回不一致的特征'''

    def __init__(self, threshold=0.05):
        self.pvalue = threshold

    def num_stab_test(self, train, test, feature=None):
        '''Compute the Kolmogorov-Smirnov statistic on 2 samples.
        检验数值特征在训练集和测试集分布是否一致,ks检验，null hypothesis是两个样本取自
        同一分布，当pvalue小于设定阈值，则拒绝原假设，则训练集和测试集的特征样本不是取自同一
        分布。可以考虑是否去除这个特征。'''
        if feature is None:
            _, pvalue = stats.ks_2samp(train, test)
        else:
            _, pvalue = stats.ks_2samp(train[feature], test[feature])
        return pvalue
    
    def num_stab_tests(self, train, test, features):
        values = [self.num_stab_test(train, test, feature) for feature in features]
        mask = [value > self.pvalue for value in values]
        return mask, values
    
    def get_value_count(self, train, test, feature=None, normalize=True):
        if feature is None:
            count = train.value_counts(normalize=normalize)
            count1 = test.value_counts(normalize=normalize)
        else:
            count = train[feature].value_counts(normalize=normalize)
            count1 = test[feature].value_counts(normalize=normalize)
        index = count.index|count1.index
        if normalize:
            count = count.reindex(index).fillna(1e-3)
            count1 = count1.reindex(index).fillna(1e-3)
        else:
            count = count.reindex(index).fillna(1)
            count1 = count1.reindex(index).fillna(1)
        return count, count1
    
    def psi(self, train, test, feature=None): # Population Stability Index
        '''PSI大于0.1则视为不太稳定。越小越稳定,通过PSI来评估特征在训练集和测试集上
        分布的稳定性。'''
        count, count1 = self.get_value_count(train, test, feature)
        res = (count1 - count)*np.log(count1/count)
        return res.sum()
    
    def psis(self, train, test, features):
        value = [self.psi(train, test, feature) for feature in features]
        res = pd.Series(value, index=features)
        return res
    
    def cat_stab_test(self, train, test, feature=None):
        '''检验类别特征在训练集和测试集分布是否一致。chi2检验，null hypothesis为分布相互
        对立，所以pvalue小于设定值拒绝原假设，即特征的分布与训练集和测试集有关，即
        特征分布在训练集和测试集不是一致的。'''
        count, count1 = self.get_value_count(train, test, feature, normalize=False)
        data = pd.concat([count,count1],axis=1)
        _, pvalue,*_ = stats.chi2_contingency(data.to_numpy().T, correction=False)
        return pvalue
    
    def cat_stab_tests(self, train, test, features):
        values = [self.cat_stab_test(train, test, feature) for feature in features]
        mask = [value > self.pvalue for value in values]
        return mask, values

    def get_labels(self, labels=None):
        if labels is None:
            label1, label2 = 'Train', 'Test'
            return label1, label2
        elif isinstance(labels, Iterable) and len(labels)>=2:
            label1, label2 = labels[0], labels[1]
            return label1, label2
        else:
            raise ValueError('labels is wrong!')
    
    def plot_train_test_num(self, train, test, features, labels=None):
        '''可视化数值特征在训练集和测试集上的分布。'''
        label1, label2 = self.get_labels(labels)
        if isinstance(features, str):
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot()
            fig.suptitle('Distribution of values in {} and {}'.format(label1, label2), fontsize=16)
            ax.set_title('Distribution of {}'.format(features))
            sns.distplot(train[features], color="green", kde=True, bins=50, label=label1, ax=ax)
            sns.distplot(test[features], color="blue", kde=True, bins=50, label=label2, ax=ax)
            plt.legend(loc=2)
        elif isinstance(features, Iterable):
            nrows = len(features)//4+1
            fig = plt.figure(figsize=(20, 5*nrows))
            fig.suptitle('Distribution of values in {} and {}'.format(label1, label2), 
                         fontsize=16, horizontalalignment='right')
            grid = gridspec.GridSpec(nrows, 4)
            for i, each in enumerate(features):
                ax = fig.add_subplot(grid[i])
                sns.distplot(train[each], color="green", kde=True, bins=50, label=label1, ax=ax)
                sns.distplot(test[each], color="blue", kde=True, bins=50, label=label2, ax=ax)
                ax.set_xlabel('')
                plt.legend(loc=2)
                ax.set_title('Distribution of {}'.format(each))
                plt.legend(loc=2)
        else:
            raise TypeError('{} is not right datatype'.format(type(features)))

    def get_melt(self, train, test, feature, labels):
        res = train[feature].value_counts(normalize=True)
        res1 = test[feature].value_counts(normalize=True)
        data = pd.concat([res,res1], axis=1).fillna(0)
        data.columns = labels
        data = data.reset_index()  # index变为column后的name默认为index
        melt = pd.melt(data, id_vars='index')
        return melt

    def plot_train_test_cat(self, train, test, features, labels=None):
        '''可视化类别特征在训练集和测试集上的分布。'''
        label1, label2 = self.get_labels(labels)
        if isinstance(features, str):
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot()
            fig.suptitle('Distribution of values in {} and {}'.format(label1, label2), fontsize=16)
            ax.set_title('{}'.format(features))
            melt = self.get_melt(train, test, features, [label1, label2])
            sns.barplot(x='index',y='value', data=melt, hue='variable', ax=ax)
            plt.legend(loc=2)
        elif isinstance(features, Iterable):
            nrows = len(features)//4 + 1
            fig = plt.figure(figsize=(20, 5*nrows))
            fig.suptitle('Distribution of values in {} and {}'.format(label1, label2), 
                         fontsize=16, horizontalalignment='right')
            grid = gridspec.GridSpec(nrows, 4)
            for i, each in enumerate(features):
                ax = fig.add_subplot(grid[i])
                melt = self.get_melt(train, test, each, [label1, label2])
                sns.barplot(x='index',y='value', data=melt, hue='variable', ax=ax)
                ax.set_title('{}'.format(each))
                plt.legend(loc=2)
        else:
            raise TypeError('{} is not right datatype'.format(type(features)))
    
    def target_split_data(self, data, target):
        '''根据目标特征对训练集进行划分。'''
        mask = data[target] == 0
        train = data.loc[mask, :]
        test = data.loc[~mask, :]
        labels = ['Target=0', 'Target=1']
        return train, test, labels
    
    def plot_target_feature_cat(self, data, features, target):
        '''可视化类别特征在目标变量上的分布。'''
        train, test, labels = self.target_split_data(data, target)
        self.plot_train_test_cat(train, test, features, labels)
    
    def plot_target_feature_num(self, data, features, target):
        '''可视化数值特征在目标变量上的分布。'''
        train, test, labels = self.target_split_data(data, target)
        self.plot_train_test_num(train, test, features, labels)
