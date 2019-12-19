from matplotlib import gridspec
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections.abc import Iterable

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号


def split_cat_num(data, cat=15):
    '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其
    归为类别特征中。'''
    categorical = data.select_dtypes(include='object')
    numerical = data.select_dtypes(exclude='object')
    nunique = numerical.nunique().sort_values(ascending=True)
    n_index = nunique[nunique < cat].index
    num = numerical.columns.difference(n_index)
    category = categorical.columns
    return category, num, n_index


class FeatureStatistics:

    def split(self, data):  # 划分类别和数值特征
        numerical = data.select_dtypes(exclude='object')
        categorical = data.select_dtypes(include='object')
        return numerical, categorical

    def describe(self, data):  # 得到数值和类别特征的一些统计特征
        numerical, categorical = self.split(data)
        if not numerical.empty:
            _ = self.describe_num(numerical)
        if not categorical.empty:
            _ = self.describe_cat(categorical)

    def _describe_all(self, data):
        length = data.shape[0]
        result = pd.DataFrame(columns=['数据名', '空值比例', '类别数', '高频类别', '高频类别比例'])
        result['数据名'] = numerical.columns
        result = self.num.set_index('数据名')
        result['空值比例'] = pd.isnull(numerical).sum()/length
        result['类别数'] = numerical.nunique()
        result['高频类别'] = numerical.apply(
            lambda x: x.value_counts().sort_values(ascending=False).index[0])
        result['高频类别比例'] = (numerical == self.num['高频类别']).sum()/length
        return result, length

    def describe_num(self, numerical, stat=False):
        '''得到数值特征的统计特征，主要为缺失值、高频值、负值比例、其他统计特征'''
        self.num, length = self._describe_all(numerical)
        self.num['负值比例'] = (numerical < 0).sum()/length
        self.num['零值比例'] = (numerical == 0).sum()/length
        if stat:
            statis = self.statistic(numerical)
            self.num = self.num.join(statis)
        self.num = self.num.reset_index()
        return self.num

    def statistic(self, numerical):  # 得到数值特征的统计特征。
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
        '''得到类别特征的信息，包括空值、高频值、熵值'''
        self.cat, _ = self._describe_all(categorical)
        self.cat['熵'] = categorical.apply(lambda x: stats.entropy(
            x.value_counts(normalize=True), base=2))/np.log2(self.cat['类别数'])  # 越接近0代表分布越集中，越接近1代表分布越分散。
        self.cat = self.cat.reset_index()
        return self.cat

    def plot(self, data):  # 可视化类别和数值特征，数值默认为分布图，类别默认为柱状图
        numerical, categorical = self.split(data)
        if not numerical.empty:
            self.plot_numerical(numerical)
        if not categorical.empty:
            self.plot_categories(categorical)

    def plot_numerical(self, numerical, style=sns.distplot):
        '''数值特征的可视化，包括箱型图，kdeplot, histplot'''
        melt = pd.melt(numerical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(style, 'value')

    def plot_categories(self, categorical):
        melt = pd.melt(categorical)
        g = sns.FacetGrid(data=melt, col="variable",
                          col_wrap=4, sharex=False, sharey=False)
        g.map(self._catplot, 'value')

    def _catplot(self, ser, **kwarg):  # 类别特征的可视化
        count = ser.value_counts()
        ax = plt.gca()
        if count.shape[0] > 50:  # 对于取值较多的特征，绘制散点图。
            ax.scatter(x=count.index, y=count.values, **kwarg)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.xaxis.set_ticklabels([])
        else:  # 其他情况下绘制条形图。
            sns.barplot(x=count.index, y=count.values, ax=ax, **kwarg)


class CatNum():
    def __init__(self):
        self.font1 = {'family': 'Calibri', 'weight': 'normal', 'size': 14}
        self.font2 = {'family': 'Calibri', 'weight': 'normal', 'size': 18}

    def get_ax(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot()
            return ax
        return ax


class Categorical(CatNum):
    '''对类别特征进行分析，适合取值不多的类别特征。对于取值很多的类别特征可视化效果不好'''

    def _crosstab(self, data, feature, target, ax=None):
        '''查看目标在类别特征的每个值的分布情况。'''
        ax = self.get_ax(ax)
        ct = pd.crosstab(data[feature], data[target], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('{}'.format(feature))
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
            self._crosstab(data, target, each, ax=ax)
        fig.subplots_adjust(wspace=0.3, hspace=0.2)
        return fig, ax

    def plot_dists(self, data, features, target):
        nrows = len(features)
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_dist(data, each, target, [ax, ax1])
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        return fig

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
        ax = self.get_ax(ax)
        df = pd.crosstab(data[feature], data[target])
        df.plot(kind='bar', stacked=True, ax=ax, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax.set_ylim(bottom=0, top=len(data))
        ax.legend(loc=2)
        xs = ax.xaxis.get_ticklocs()
        odds = df[1]/df[0]
        axt = ax.twinx()
        axt.plot(xs, odds, marker="o")
        axt.set_ylabel("odds", fontdict=self.font1, labelpad=6)
        odd = df.sum()[1]/df.sum()[0]
        axt.axhline(odd, color="crimson", alpha=0.9, linestyle="--")
        return ax


class Numerical(CatNum):

    def drop_null_item(self, data, feature=None):
        if feature is None:
            return data[data.notnull()]
        temp = data.loc[data[feature].notnull(), feature]
        return temp

    def _kstest(self, data, feature):  # 检验特征分布是否为正态分布。
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

    def plot_kde(self, data, feature, ax=None):
        ax = self.get_ax(ax)
        sample = self.drop_null_item(data, feature).to_numpy()
        dist = stats.kde.gaussian_kde(sample)
        sample.sort()
        ax.plot(sample, dist.pdf(sample), label='kde')
        mean, std = sample.mean(), sample.std()
        ax.plot(sample, stats.norm(mean, std).pdf(
            sample), color='k', label='norm')
        ax.set_title('{} kde distribution'.format(
            feature), fontdict=self.font1)
        ax.set_ylabel('Probability', fontdict=self.font2, labelpad=6)
        ax.legend(loc=2)
        return ax

    def plot_kdes(self, data, features):  # 绘制kde图
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            self.plot_kde(data, each, ax=ax)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        return fig

    def plot_boxs(self, data, features):  # 绘制箱型图
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            _ = self.plot_box(self, data, each, ax=ax)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        return fig

    def plot_box(self, data, feature, ax=None):
        ax = self.get_ax(ax)
        sns.boxplot(data[each], ax=ax, orient='v')
        ax.set_title('{} boxplot'.format(each), fontdict=self.font2)
        ax.set_ylabel('{}'.format(each), fontdict=self.font1, labelpad=6)
        return ax

    def plot_hists(self, data, features, target):
        nrows = len(features)
        fig = plt.figure(figsize=(8, 4*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            self.plot_hist(data, each, ax=ax)
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_hist_line(data, each, target, ax=ax1)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        return fig

    def plot_hist(self, data, feature, ax=None):
        ax = self.get_ax(ax)
        temp = self.drop_null_item(data, feature)
        bins = int(np.sqrt(temp.shape[0]))
        _ = ax.hist(data[feature], bins=bins, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_xlabel('{}'.format(feature), fontdict=self.font1, labelpad=2)
        ax.set_ylabel('frequency', fontdict=self.font1, labelpad=6)
        return ax

    def plot_hist_line(self, data, feature, target, ax=None):
        '''调用 ax.hist 创建直方图，会返回3个返回值，第一个返回值n就是一个向量，
        记录了这个直方图的每个柱子的高度。而想做叠加的直方图，可以再使用一条 ax.hist语句，将bottom设置为上一个图返回的n。
        而bins返回的是每个区间的端点，是一个数组，共含(n+1)个值。'''
        ax = self.get_ax(ax)
        X0 = data.loc[data[target] == 0, feature]
        X1 = data.loc[data[target] == 1, feature]
        bins = int(np.sqrt(min(len(X0), len(X1))))
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

class PlotFunc:

    def get_ax(slef, ax=None):
        if ax is None:
            fig = plt.figure(figsize(8,5))
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

        ax.set_title('Missing Rate', fontdict={'size': 22})
        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'], rotation=90,
                           fontdict={'horizontalalignment': 'right', 'size': 12})

        for row in data.itertuples():
            ax.text(row.Index, row.b*1.01, s=round(row.b, 1), horizontalalignment='center',
                    verticalalignment='bottom', fontsize=14)
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
        records_count = data.shape[0]
        col_most_values, col_large_value = {}, {}

        for col in data.columns:
            value_counts = data[col].value_counts()
            col_most_values[col] = value_counts.max()/records_count
            col_large_value[col] = value_counts.idxmax()

        most_values_df = pd.DataFrame.from_dict(
            col_most_values, orient='index')  # 字典的key为index
        most_values_df.columns = ['max percent']
        most_values_df = most_values_df.sort_values(
            by='max percent', ascending=False)
        return most_values_df, col_large_value

    def plot_frequency(self, data, N=30):
        '''将样本中有缺失的特征的缺失率按从大到小绘制出来'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号
        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylabel('Frequency Rate')
        ser, _ = self.most_frequent(data)
        ser = ser[:N]
        data = ser.reset_index()
        self.plot_bin(data, ax=ax)

    def frequence_bin(self, data, features):
        '''对特征进行0-1编码的特征，出现次数最多的的样本为一类，其他的为一类'''
        result = data[features].copy()
        for each in features:
            col_large = result[each].value_counts().idxmax()
            result[each+'_bins'] = (result[each] == col_large).astype(int)
        return result

    def delete_frequency(self, data, threshold=None):
        '''对于没有明显差异，同时特征中某个值出现的概率很高的特征进行删除。 '''
        if threshold is None:
            threshold = self.deletes
        result = data.copy()
        col_most, _ = self.most_frequent(result)
        large_percent_cols = list(
            col_most[col_most['max percent'] >= threshold].index)
        result = result.drop(large_percent_cols, axis=1)
        return result, large_percent_cols

    def fit(self, X, y=None):
        col_most, col_large_value = self.most_frequent(X)
        large_percent_cols = list(
            col_most[col_most['max percent'] >= self.deletes].index)
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
        large_percent_cols = list(
            col_most[col_most['max percent'] >= self.deletes].index)
        self.large_percent_ = large_percent_cols
        self.col_large_ = col_large_value
        result = result.drop(large_percent_cols, axis=1)
        return result


class CalEnt:
    '''比较某字段在类别合并前后的信息熵、基尼值、信息值IV，若合并后信息熵值减小/基尼值减小/信息值IV增大，
    则确认合并类别有助于提高此字段的分类能力，可以考虑合并类别。'''
    def _entropy(self, group):
        return -group*np.log2(group+1e-5)

    def _gini(self, group):
        return group*(1-group)

    def cal(self, data, feature, target, func=None):
        '''计算使用的通用函数'''
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = func(temp)
        return (temp1*enti.sum(axis=1)).sum()
         
    def entropy(self, data, feature, target):
        # 熵值Ent(D)越小，表示变量的分类能力越强，即预测能力越强。
        if isinstance(feature, str):
            return self.cal(data, feature, target, self._entropy)
        elif isinstance(feature, Iterable):
            return [self.cal(data, each, target, self._entropy) for each in feature]
        else:
            raise TypeError('Feature is not right data type')

    def gini(self, data, feature, target):
        # 基尼值越小，说明数据集的纯度越高，即变量的预测力越强。
        if isinstance(feature, str):
            return self.cal(data, feature, target, self._gini)
        elif isinstance(feature, Iterable):
            return [self.cal(data, each, target, self._gini) for each in feature]
        else:
            raise TypeError('Feature is not right data type')

    def woe_iv(self, data, feature, target): # 计算WOE, IV值
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = np.log((temp.iloc[:, 0]+1e-5)/(temp.iloc[:, 1]+1e-5))
        iv = (temp.iloc[:, 0] - temp.iloc[:, 1])*woei
        return iv.sum(), woei.to_dict()

    def woe_ivs(self, data, features, target):
        if isinstance(features, str):
            return self.woe_iv(data, feature, target)
        elif isinstance(features, Iterable):
            return{each: self.woe_iv(data, each, target) for each in features}
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
        self.plot_bin(data, ax=ax)

    def plot_item_miss(self, data):
        '''将每个样本的缺失值个数按从小到大绘制出来。'''
        ser = (data.isnull().sum(axis=1)).sort_values()
        x = range(data.shape[0])
        plt.scatter(x, ser.values, c='black')

    def find_index(self, data, threshold=None):
        '''找到满足条件的特征'''
        length = data.shape[0]
        _, null_sum, _ = self.is_null(data)
        ratio = null_sum()/length
        index = ratio[ratio >= threshold].index
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
        result = data.copy()
        *_, null_item = self.is_null(data)
        index2 = null_item[null_item > value].index
        result = result.drop(index2)
        return result

    def indicator_null(self, data, threshold=None):
        '''生产特征是否为缺失的指示特征，同时删除原特征。'''
        result = data.copy()
        if threshold is None:
            threshold = self.indicator
        index = self.find_index(data, threshold)
        for each in index:
            result['is_null_'+each] = pd.isnull(result[each]).astype(int)
        result = result.drop(index, axis=1)
        return index, result

    def another_class(self, data, features):
        '''对于特征而言，可以将缺失值另作一类'''
        result = data.loc[:, features].copy()
        fill_value = {}
        for each in features:
            if data[each].dtype == 'object':
                fill_value[each] = 'None'
            else:
                fill_value[each] = int(data[each].max()+1)
        result.fillna(self.fill_value_)
        return fill_value, result

    def bin_and_fill(self, data, features):
        '''对于一些数值特征，我们可以使用中位数填补，但是为了不丢失缺失信息，同时可以进行编码。'''
        result = data.loc[:, features].copy()
        fill_value = {}
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



class Outlier:
    def __init__(self, method=sns.stripplot):
        self.method = method

    def plot_num_data(self, data, target, features=None):
        '''按照label,对数据集中的数值特征绘制stripplot，可以根据图形从中寻找到
        数值特征中的异常值。'''
        if features is None:
            _, num, _ = split_cat_num(data)
        else:
            num = features
        data = pd.concat([data, target], axis=1)
        name = target.name
        melt = pd.melt(data, id_vars=name,
                       value_vars=[f for f in num])
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4,
                          sharex=False, sharey=False)
        g.map(self.method, name, 'value', jitter=True,
              palette="muted", order=[0, 1])

    def plot_cat_data(self, data, y, features=None):
        if features is None:
            cat, _, n_index = split_cat_num(data)
            cols = cat.union(n_index)
        else:
            cols = features
        self.ploting_cat_fet(data, cols, y)

    def ploting_cat_fet(self, df, cols, y):
        '''绘制类别特征，柱状图为每个值在特征中出现的数目及所占的比例，折线图为每个取值
        情况下，其中的坏样本率。'''
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号
        total = len(df)
        # 图形的参数设置
        nrows, ncols = len(cols)//2+1, 2
        grid = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(16, 20*nrows//3))
        df = pd.concat([df, y], axis=1)
        name = y.name

        for n, col in enumerate(cols):
            tmp = pd.crosstab(df[col], df[name],
                              normalize='index') * 100
            tmp = tmp.reset_index()
            tmp.rename(columns={0: 'No', 1: 'Yes'}, inplace=True)

            ax = fig.add_subplot(grid[n])
            sns.countplot(x=col, data=df, order=list(tmp[col].values),
                          color='green')  # 绘制柱状图

            ax.set_ylabel('Count', fontsize=12)  # 设置柱状图的参数
            ax.set_title(f'{col} Distribution by Target', fontsize=14)
            ax.set_xlabel(f'{col} values', fontsize=12)

            gt = ax.twinx()  # 绘制折线图
            gt = sns.pointplot(x=col, y='Yes', data=tmp,
                               order=list(tmp[col].values),
                               color='black', legend=False)

            mn, mx = gt.get_xbound()  # 绘制水平线
            gt.hlines(y=y.sum()/total, xmin=mn, xmax=mx,
                      color='r', linestyles='--')

            gt.set_ylim(0, tmp['Yes'].max()*1.1)  # 设置y轴和title
            gt.set_ylabel("Target %True(1)", fontsize=16)
            sizes = []

            for p in ax.patches:  # 标识每个值所占的比例。
                height = p.get_height()
                sizes.append(height)
                ax.text(p.get_x()+p.get_width()/2.,
                        height + 3,
                        '{:1.2f}%'.format(height/total*100),
                        ha="center", fontsize=12)
            ax.set_ylim(0, max(sizes) * 1.15)
        plt.subplots_adjust(hspace=0.5, wspace=.3)