'''
这个模块主要包括一些计算相关系数，和进行特征选择的类。
Relation: 为计算相关系数的抽象基类，所有计算相关系数的类别都将继承这个类，子类必须实现
ftrelation和ffrelation两个方法，分别计算特征与目标之间的相关系数和特征之间的相关系数。
同时也包含可视化特征系数的函数。

Pearson:基于Relation实现的pearson相关系数
MICRelation:基于Relation实现的最大互信息数
DistRelation:基于Relation实现的距离相关系数

FSelect:进行特征选择的抽象基类，所有进行特征选择的类都将继承自这个类，所有子类必须实现
filtermask这个函数，此函数产生特征掩码，来进行特征选择。

FilterSelect：基于FSelect实现的过滤式特征选择方法，同样也是一个抽象基类，所有的过滤式特征
选择的类都将继承自此类，子类必须实现get_importance这个函数，此函数用于评价特征的重要程度，是
进行过滤式特征选择的基础。
CoreSelect：一个基kendall相关系数的过滤式特征选择方法。

WrapperSelect：基于FSelect实现的包裹式特征选择方法。所有基于包裹式的特征选择方法都必须为它的子类，
所有子类必须实现choose函数，这个函数是包裹式特征选择如何选出最优的特征子集的实现函数，返回结果可以是
所有特征的一个子集，也可以是掩码。
FeatureImportanceSelect：基于后向选择的包裹式特征选择方法，不断从特征集中剔除掉不重要的特征。

'''


import minepy
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn import clone
from DataPreProcess import PlotFunc
import matplotlib.pyplot as plt
from abc import abstractmethod
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.spatial.distance import pdist, squareform


class Relation(PlotFunc):
    '''计算相关系数的抽象基类。'''
    def pd2array(self, X):
        '''将DataFrame转化为ndarray，同时记录下columns和columns长度。'''
        columns = X.columns
        arr = X.to_numpy()
        return arr, columns, X.shape[1]

    def computeft(self, X, y, func):
        '''计算特征变量和目标变量之间的相关系数。比较常用，所以实现了'''
        arr, columns, n = self.pd2array(X)
        yarr = y.to_numpy()
        data = np.fromiter((func(arr[:, j], yarr)[0] for j in range(n)),
                           dtype=np.float32)
        result = pd.Series(data, index=columns)
        return result

    def computeff(self, X, func):
        '''计算特征变量与特征变量之间的相关系数。比较常用所以实现。'''
        arr, columns, n = self.pd2array(X)
        try:
            result, _ = func(X)
        except TypeError:
            result = [
                func(arr[:, i], arr[:, j])[0] for i in range(n - 1)
                for j in range(i + 1, n)
            ]
            result = squareform(result) + np.identity(n)
        result = pd.DataFrame(result, columns=columns, index=columns)
        return result

    @abstractmethod
    def ftrelation(self, X, y):  # 计算特征变量与目标变量之间的相关系数
        pass

    @abstractmethod
    def ffrelation(self, X):  # 计算特征变量之间的相关系数。
        pass

    def relation(self, X, y=None):  # 计算相关系数的接口。
        if y is None:
            return self.ffrelation(X)
        else:
            return self.ftrelation(X, y)

    def plot_ft(self, data):
        '''可视化特征变量和目标变量之间的相关系数，也可以用来可视化Series序列。'''
        data = data.sort_values().reset_index()  # 特征与label之间的相关性的可视化。
        self.plot_bin(data)

    def plot_ff(self, data):
        '''用热力图来可视化特征变量与特征变量之间的相关系数。也可以用来绘制其他的热力图。'''
        _, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data,
                    xticklabels=data.columns,
                    yticklabels=data.columns,
                    cmap='RdYlGn',
                    center=0,
                    annot=True,
                    ax=ax)
        ax.set_title('Correlogram Heatmap')

    def plot_relation(self, X, y=None):
        '''可视化相关系数的接口函数。'''
        if y is None:
            data = self.ffrelation(X)
            self.plot_ff(data)
        else:
            data = self.ftrelation(X, y)
            self.plot_ft(data)


class Pearson(Relation):  # 皮尔逊相关系数。
    def ftrelation(self, X, y):
        return self.computeft(X, y, func=stats.pearsonr)

    def ffrelation(self, X):
        return self.computeff(X, func=stats.pearsonr)


class MICRelation(Relation):  # 最大信息系数。
    def ftrelation(self, X, y):
        arr, columns, _ = self.pd2array(X)
        arr = arr.transpose(1, 0)
        yarr = y.to_numpy()[None, :]
        result, _ = minepy.cstats(arr, yarr)
        result = pd.Series(result.ravel(), index=columns)
        return result

    def ffrelation(self, X):
        arr, columns, n = self.pd2array(X)
        arr = arr.transpose(1, 0)
        mic = squareform(minepy.pstats(arr)[0])+np.identity(n)
        result = pd.DataFrame(mic, columns=columns, index=columns)
        return result


class DistRelation(Relation):  # 距离相关系数。

    def _distcorr(self, X, Y):
        X, Y, n = X[:, None], Y[:, None], X.shape[0]
        a, b = squareform(pdist(X)), squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return (dcor, None)

    def ftrelation(self, X, y):
        return self.computeft(X, y, func=self._distcorr)

    def ffrelation(self, X):
        return self.computeff(X, func=self._distcorr)


class FSelection:
    '''进行特征选择的抽象类。'''

    @abstractmethod
    def filtermask(self, X, y=None, **kwargs):
        '''产生关于特征的0-1掩码，0表示特征没有被选择， 1表示特征被选择。'''
        pass

    def getnumber(self, X, to_select=None):
        '''在没有给定选择的特征数目的情况下，选择删除一般的特征。'''
        if to_select is None:
            return X.shape[1]//2
        return to_select

    def fit(self, X, y=None, **kwargs):
        mask = self.filtermask(X, y, **kwargs)
        self.mask = list(map(bool, mask))
        return self

    def transform(self, X):
        result = X.copy()
        return result.loc[:, self.mask]

    def fit_transform(self, X, y, **kwargs):
        _ = self.fit(X, y, **kwargs)
        result = self.transform(X)
        return result


class FilterSelect(FSelection):
    '''Filter特征选择方式，作为一种与模型无关的选择方式，首先得到每个特征的得分，根据得分的高低
    来表示特征对于建模的重要性，然后选择其中重要的一部分进行建模。'''
    @abstractmethod
    def get_importance(self, X, feature, y=None):
        '''得到每个特征的重要性。'''
        pass

    def get_importances(self, X, y=None, isabs=True):
        '''得到特征的重要性。isabs确定是否是值的绝对值大小与特征重要程度有关，符号无关。'''
        features = X.columns
        scores = [self.get_importance(X, feature, y) for feature in features]
        if isabs:
            return np.fabs(scores)
        else:
            return np.array(scores)

    def filtermask(self,
                   X,
                   y=None,
                   ascending=False,
                   to_select=None,
                   isabs=True):
        '''根据每个特征的得分，产生0-1掩码，ascending参数为True表示得分越高越重要，False表示
        得分越低越重要。to_select确定特征选择后的特征数量，如果为None则保留一般的特征，isabs是确定
        正负号对特征重要性的影响，如果知识绝对值的大小决定特征的重要性，则为True。'''
        n = self.getnumber(X, to_select)
        scores = self.get_importances(X, y, isabs)
        mask = np.zeros(X.shape[1])
        index = scores.argsort()
        if ascending:
            select = index[-n:]
        else:
            select = index[:n]
        mask[select] = 1
        return mask

    def plot_importances(self,
                         X,
                         y=None,
                         ascending=False,
                         to_select=None,
                         isabs=True):
        '''可视化特征的重要程度，被选中的特征为红色， 没有被选择的特征为蓝色。'''
        mask = self.filtermask(X, y, ascending, to_select, isabs)

        scores = self.get_importances(X, y, isabs)
        score = pd.Series(scores, name='score', index=X.columns)
        colors = ['red' if ma else 'blue' for ma in mask]
        color = pd.Series(colors, name='color', index=X.columns)

        data = pd.concat([score, color], axis=1)
        data.index.name = 'index'
        data.reset_index(inplace=True)

        fig = plt.figure(figsize=(9, 6), dpi=80)
        ax = fig.add_subplot(111)

        ax.vlines(x=data.index,
                  ymin=0,
                  ymax=data['score'],
                  color=data['color'],
                  alpha=0.4,
                  linewidth=5)

        for x, y, tex, color in zip(data.index, data['score'], data['score'],
                                    data['color']):
            _ = plt.text(x,
                         y,
                         round(tex, 3),
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontdict={
                             'color': color,
                             'size': 8
                         })

        ax.set(ylabel='$Value$', xlabel='$Columns$')
        plt.xticks(data.index, data['index'], fontsize=9, rotation=90)
        ax.set_title('Importances Graph', fontdict={'size': 18})
        ax.grid(linestyle='--', alpha=0.5)


class CoreSelect(FilterSelect):
    '''子类中可以重新改写filtermask，从而修改默认参数，使得在fit时减少参数的输入，也可以在fit时设定参数。'''
    def get_importance(self, X, feature, y):
        value = stats.kendalltau(X[feature], y)[0]
        return value

    def filtermask(self, X, y=None, ascending=True, to_select=None,
                   isabs=True):
        result = super().filtermask(X, y, ascending, to_select, isabs)
        return result


fs = CoreSelect()
# fs.fit(train, y)
# 也可以不在子类中重新改写filtermask，而在fit时设定参数
# fs.fit(train, y, ascending=True)


class WrapperSelect(FSelection):
    def __init__(self, clf, scoring, n=5, imbalance=False):
        '''Wrapper式特征选择，格局模型选择最适合模型的特征子集，初始化参数包括，模型、评价函数、
        是否为类别不平衡。'''
        self.clf = clf  # 模型
        self.scoring = scoring  # 评价函数
        self.n = n
        if imbalance:
            self.sp = StratifiedKFold(n, shuffle=True, random_state=10)
        else:
            self.sp = KFold(n, shuffle=True, random_state=10)

    def get_score(self, X, y, subfeatures=None):
        '''得到每个子集在评价函数下的得分均值和方差。'''
        if subfeatures is None:
            subfeatures = X.columns
        scores = np.zeros(self.n)
        for i, (tidx, vidx) in enumerate(self.sp.split(X, y)):
            trainX, vlidX = X.loc[tidx, subfeatures], X.loc[vidx, subfeatures]
            trainy, vlidy = y[tidx], y[vidx]
            est = self.clf.fit(trainX, trainy)
            pred = est.predict_proba(vlidX)
            scores[i] = self.scoring(vlidy, pred[:, 1])
        return scores.mean(), scores.std()

    @abstractmethod
    def choose(self, X, y, to_select=None):
        '''得到最优特征子集的函数。'''
        pass

    def filtermask(self, X, y):
        '''根据得到的特征子集产生0-1掩码。'''
        result = self.choose(X, y)
        if (set(result) == {0, 1}) and len(result) == X.shape[1]:
            return result
        elif set(result) <= set(X.columns):  # 子集
            mask = np.zeros(X.shape[1])
            index = [
                i for i, feature in enumerate(X.columns) if feature in result
            ]
            mask[index] = 1
            return mask
        else:
            raise ValueError('子类实现的方法返回值不满足:[0,1]序列或特征的子集的条件。')


class FeatureImportanceSelect(WrapperSelect):
    def get_thres(self, X, y):
        '''得到不同特征的重要程度和最小值。'''
        est = clone(self.clf)
        est.fit(X.to_numpy(), y)
        if hasattr(est, 'coef_'):
            importance = est.coef_
        elif hasattr(est, 'feature_importances_'):
            importance = est.feature_importances_
        else:
            raise AttributeError(
                '{} do not has coef_ or feature_importance_'.format(
                    self.clf.__class__.__name__))
        return importance, importance.min()

    def choose(self, X, y, to_select=None):
        '''不断出去最不重要的特征，最终选取结果最好的作为最终选择的特征进行后续建模。'''
        n = self.getnumber(X, to_select)
        result = X.copy()
        mean, std = self.get_score(result, y)
        means = [mean]
        stds = [std]
        maxi = means[0]
        subfeatures = result.columns
        while result.shape[1] > n:
            importance, thres = self.get_thres(result, y)
            mask = importance > thres
            result = result.loc[:, mask].copy()
            mean, std = self.get_score(result, y)
            means.append(mean)
            stds.append(std)
            if mean > maxi-0.001:
                maxi = mean
                subfeatures = result.columns
        self.means = np.array(means)
        self.stds = np.array(stds)
        return subfeatures

    def plot_process(self):
        '''可视化在不断去除掉最不重要特征的情况下，结果的变化过程。'''
        x = range(len(self.means))
        ax = plt.subplot(1, 1, 1)
        ax.plot(x, self.means, color='red', lw=2)
        ax.fill_between(x,
                        self.means + self.stds,
                        self.means - self.stds,
                        alpha=0.3)
        ax.scatter(self.means.argmax(),
                   self.means.max(),
                   color='black',
                   s=50,
                   edgecolors='purple')
        ax.vlines(self.means.argmax(), 0, self.means.max())
        ax.axhline(self.means.max(), linestyle='--', color='c')
        ax.text(self.means.argmax(),
                self.means.max() * 1.01, '{:.3f}'.format(self.means.max()))
        ax.set_ylim(self.means.min() * 0.9, self.means.max() * 1.1)
