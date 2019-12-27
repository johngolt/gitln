from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import minepy
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import RFECV, GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier
from abc import abstractmethod
from sklearn import clone


class Relation:

    def pd2array(self, X):
        columns = X.columns
        values = X.to_numpy()
        return values, columns, X.shape[1]

    def computeft(self, X, y, func):
        columns, arr, n = self.pd2array(X)
        yarr = y.to_numpy()
        data = np.array(func(arr[:,j], yarr)[0] for j in range(n))
        result = pd.Series(data, index=columns)
        return result

    def computeff(self, X, func):
        columns, arr, n = self.pd2array(X)
        try:
            result, _ = func(X)
        except:
            result = [func(arr[:,i], arr[:,j])[0] for i in range(n-1) for j in range(i+1, n)]
            result = squareform(result) + np.identity(n)
        result = pd.DataFrame(result, columns=columns, index=columns)
        return result

    @abstractmethod
    def ftrelation(self, X, y):
        pass

    @abstractmethod
    def ffrelation(self, X):
        pass

    def relation(self, X, y=None):
        if y is None:
            return self.ffrelation(X)
        else:
            return self.ftrelation(X, y)

    def plot_ft(self, data):
        _, ax = plt.subplots(figsize=(8,5))
        data=data.sort_values().reset_index()  # 特征与label之间的相关性的可视化。
        data.columns=['a','b']
        ax.vlines(x=data.index, ymin=0, ymax=data['b'], 
                    color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'], s=75, 
                    color='firebrick', alpha=0.7)
        ax.set_title('Correlogram', fontdict={'size':22})
        ax.set_xticks(data.index)
        mn,mx = ax.get_xbound()
        ax.hlines(y=0,xmin=mn,xmax=mx,linestyles='--')
        ax.set_xticklabels(data['a'].str.title(), rotation=60, 
                          fontdict={'horizontalalignment': 'right', 'size':12})
        for row in data.itertuples():
            ax.text(row.Index, row.b*1.01, s=round(row.b, 2),
                    horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

    def plot_ff(self, data):
        _, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(data, xticklabels=data.columns, yticklabels=data.columns, cmap='RdYlGn',
            center=0, annot=True,ax=ax)
        ax.set_title('Correlogram Heatmap')
    
    def plot_relation(self, X, y=None):
        if y is None:
            data = self.ffrelation(X)
            self.plot_ff(data)
        else:
            data = self.ftrelation(X, y)
            self.plot_ft(data)

class Pearson(Relation):
    def ftrelation(self, X, y):
        return self.computeft(X, y, func=stats.pearsonr)
    def ffrelation(self, X):
        return self.computeff(X, func=stats.pearsonr)

class MICRelation(Relation):
    def ftrelation(self, X, y):
        columns, arr, _ = self.pd2array(X)
        arr=arr.transpose(1,0)
        yarr = y.to_numpy()[None,:]
        result,_ = minepy.cstats(arr,yarr)
        result = pd.Series(result.ravel(), index=columns)
        return result

    def ffrelation(self, X):
        columns, arr, n = self.pd2array(X)
        arr = arr.transpose(1,0)
        mic = minepy.pstats(arr)+np.identity(n)
        result = pd.DataFrame(mic, columns=columns, index=columns)
        return result

class DistRelation(Relation):

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
        self.mask = self.filtermask(X, y, **kwargs)
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

    def filtermask(self, X, y=None, ascending=False, to_select=None):
        '''根据每个特征的得分，产生0-1掩码，ascending参数为True表示得分越高越重要，False表示
        得分越低越重要。'''
        features = X.columns
        n = self.getnumber(X, to_select)
        scores = [self.get_importance(X, feature, y) for feature in features]
        mask = np.zeros(X.shape[1])
        index = np.array(scores).argsort()
        if ascending:
            select = index[-n:]
        else:
            select = index[:n]
        mask[select] = 1
        return mask 
        

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
        for i,(tidx, vidx) in enumerate(self.sp.split(X, y)):
            trainX, vlidX = X.loc[tidx, subfeatures], X.loc[vidx, subfeatures]
            trainy, vlidy = y[tidx], y[vidx]
            est = self.clf.fit(trainX, trainy)
            pred = est.predict_proba(vlidX)
            scores[i] = self.scoring(vlidy, pred[:,1])
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
        elif set(result) <= set(X.columns):
            mask = np.zeros(X.shape[1])
            index = [i for i, feature in enumerate(X.columns) if feature in result]
            mask[index] = 1
            return mask
        else:
            raise ValueError('子类实现的方法返回值不满足:[0,1]序列或特征的子集的条件。')

class ImportanceSelect(WrapperSelect):
    def get_thres(self, X, y):
        est = clone(self.clf)
        est.fit(X.to_numpy(), y)
        if hasattr(est, 'coef_'):
            importance = est.coef_
        elif hasattr(est, 'feature_importance_'):
            importance = est.feature_importance_
        else:
            raise AttributeError('{} do not has coef_ or feature_importance_'.format(
                                                   self.clf.__class__.__name__))
        thres = np.array(set(importance))
        thres.sort()
        return importance, thres[0]

    def choose(self, X, y, to_select=None):
        n = self.getnumber(X, to_select)
        result = X.copy()
        mean, std = self.get_score(result, y)[0]
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
            if mean > maxi:
                maxi = mean
                subfeatures = result.columns
        return subfeatures, means, stds
    
    def plot_process(self, mean, std):
        x = range(len(mean))
        ax = plt.subplot(1,1,1)
        ax.plot(x,mean,color='red',lw=2)
        ax.fill_between(x, mean+std, mean-std, alpha=0.3)
        ax.scatter(mean.argmax(),mean.max(),color='black',s=50,
                edgecolors='purple')
        ax.vlines(mean.argmax(),0,mean.max())
        ax.text(mean.argmax(),mean.max()*1.01,'{:.2f}'.format(mean.max()))
        ax.set_ylim(mean.min()*0.9,mean.max()*1.1)