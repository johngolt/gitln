from sklearn.model_selection import train_test_split, StratifiedKFold
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


'''Compute the first Wasserstein distance between two 1D distributions.'''
#stats.wasserstein_distance()#数值特征，分类，返回值为距离
'''Compute the energy distance between two 1D distributions.'''
#stats.energy_distance()# 数值特征分类，返回值为距离


def decorate(f): #装饰器，使得可以使用DataFrame。
    def inner(self, features, target=None):
        if isinstance(features,np.ndarray):
            result = f(self, features, target)
        else:
            self.columns = features.columns
            if target is None:
                result = f(self, features.values, target)
            else:
                result = f(self, features.values, target.values)
        return result
    return inner

class FeatureRelation:
    def __init__(self):
        self.columns = None

    def compute_X_y(self, X, y, func=None):
        _, n = X.shape
        data = np.array(func(X[:,j], y)[0] for j in range(n))
        result = pd.Series(data, index=self.columns)
        return result

    def compute_X(self, X, func=None):
        _, n = X.shape
        try:
            result, _ = func(X)
        except:
            result = []
            for i in range(n-1):
                for j in range(i+1,n):
                    value, _ = func(X[:,i], X[:,j])
                    result.append(value)
            result = squareform(result) + np.identity(n)
        result = pd.DataFrame(result, columns=self.columns, index=self.columns)
        return result
    
    @decorate
    def pearson(self, X, y=None): # 皮尔逊相关系数
        if y is None:
            return self.compute_X(X, stats.pearsonr)
        return self.compute_X_y(X,y, stats.pearsonr)
    
    @decorate
    def spearman(self, X, y=None): #斯皮尔曼相关系数
        if y is None:
            return self.compute_X(X, stats.spearmanr)
        return self.compute_X_y(X,y, stats.spearmanr)
    
    @decorate
    def kendall(self,X, y=None): # Kendall相关系数
        if y is None:
            return self.compute_X(X, stats.kendalltau)
        return self.compute_X_y(X, y, stats.kendalltau)
    
    @decorate
    def mic(self, X, y=None): # 最大信息系数
        X=X.transpose(1,0)
        if y is None:
            mic = minepy.pstats(X)+np.identity(X.shape[1])
            result = pd.DataFrame(mic, columns=self.columns, index=self.columns)
            return result
        y = y[None,:]
        result,_ = minepy.cstats(X,y)
        result = pd.Series(result.ravel(), index=self.columns)
        return result
    
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
    
    @decorate
    def distcorr(self, X, y=None):  #距离相关系数,适用于数值特征
        if y is None:
            return self.compute_X(X, self._distcorr)
        return self.compute_X_y(X, y, self._distcorr)
    
    def plot_corr(self, data):  #相关系数的可视化
        _, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.set_ylabel('Correlogram')
        dim = data.values.ndim
        assert dim in (1,2)
        if dim == 2: # heatmap的可视化
            sns.heatmap(data, xticklabels=data.columns, yticklabels=data.columns, cmap='RdYlGn',
            center=0, annot=True,ax=ax)
        else: 
            data=data.sort_values().reset_index()# 特征与label之间的相关性的可视化。
            data.columns=['a','b']
            ax.vlines(x=data.index, ymin=0, ymax=data['b'], 
                      color='firebrick', alpha=0.7, linewidth=2)
            ax.scatter(x=data.index, y=data['b'], s=75, 
                       color='firebrick', alpha=0.7)
            ax.set_title('Correlogram', 
                         fontdict={'size':22})
            ax.set_xticks(data.index)
            mn,mx = ax.get_xbound()
            ax.hlines(y=0,xmin=mn,xmax=mx,linestyles='--')
            ax.set_xticklabels(data['a'].str.upper(), 
                    rotation=60, 
                fontdict={'horizontalalignment': 'right', 'size':12})
            for row in data.itertuples():
                ax.text(row.Index, row.b*1.01, s=round(row.b, 2),
                        horizontalalignment= 'center',
                        verticalalignment='bottom', fontsize=14)

class FeatureSelection:
    def __init__(self,estimator=None, to_select=None, score='roc_auc'):
        if estimator is None: #用于进行wrapper方法的模型
            self.estimator = RandomForestClassifier(n_estimators=10, max_depth=4)
        else:
            self.estimator = estimator
        self.score = score #评估标准
        self.masked = None #得到的mask掩码，用于测试集
        
    def select_by_filter(self, func, X, y):
        # filter方法进行可视化。
        to_select = X.shape[1]//2
        gus = GenericUnivariateSelect(func, 'k_best', param=to_select)
        gus.fit(X,y)
        self.masked = gus.get_support( )
    
    def select_by_wrapper(self, X, y):
        step = max(1,X.shape[1]//50)
        to_select = X.shape[1]//2
        clf = RFECV(self.estimator,step=step,
                min_features_to_select=to_select, scoring=self.score,
                   cv=5, n_jobs=-1)
        clf.fit(X,y)
        self.masked = clf.support_
        
    def __call__(self, X, y=None,func=None):
        if y is None:
            return X.loc[:,self.masked]
        else:
            '''如果提供了评估特征与label关系的函数则采用filter，否则wrapper'''
            if func is None:
                self.select_by_wrapper(X, y)
                return X.loc[:,self.masked]
            else:
                self.select_by_filter(func,X,y)
                return X.loc[:,self.masked]

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
def each_feature_select(rf, features, y, scoring):
    names = features.columns
    X,Y = features.values, y.values
    scores={}
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring=scoring,
                              cv=ShuffleSplit(5, .3))
        scores[names[i]] = round(np.mean(score),3)
    return scores

from collections import defaultdict
def average_decrease(rf, features, y, scoring):
    names = features.columns
    X, Y = features.values,y.values
    scores = defaultdict(list)
    for train_idx, test_idx in ShuffleSplit(10, .3).split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        rf = rf.fit(X_train, Y_train)
        acc = scoring(Y_test, rf.predict_proba(X_test)[:,1])
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = scoring(Y_test, rf.predict_proba(X_t)[:,1])
            scores[names[i]].append((acc-shuff_acc)/acc)
    return sorted([(round(np.mean(score), 4), feat) for feat,
              score in scores.items()], reverse=True)


def selectwithfeatureimportance(estimator, X, Y):
    kf = StratifiedKFold(10)
    model = estimator()
    model.fit(X, Y)
    thresholds = np.sort(model.feature_importances_)
    result = np.zeros(shape=(len(thresholds),10))
    support = None
    baseline = -np.inf
    for i, thresh in enumerate(thresholds):
        score = 0
        selection = SelectFromModel(model, threshold=thresh, 
                                    prefit=True)
        for j,(tidx, vidx) in enumerate(kf.split(X, Y)): #cv计算均值和方差
            X_train, X_valid = X.iloc[tidx, :], X.iloc[vidx, :]
            y_train, y_valid = Y.iloc[tidx], Y.iloc[vidx]
            select_X_train = selection.transform(X_train)
            selection_model = estimator()
            selection_model.fit(select_X_train, y_train)
            select_X_test = selection.transform(X_valid)
            y_pred = selection_model.predict_proba(select_X_test)[:,1]
            score += roc_auc_score(y_valid,y_pred)/5
            result[i,j] = roc_auc_score(y_valid,y_pred)

        if score > baseline:
            support = selection.get_support()
            baseline = score

    return support, result, baseline

def plot_process(result):
    x,mean,std = range(result.shape[0]),result.mean(axis=1),result.std(axis=1)
    ax = plt.subplot(1,1,1)
    ax.plot(x,mean,color='red',lw=2)
    ax.fill_between(x, mean+std, mean-std, alpha=0.3)
    ax.scatter(mean.argmax(),mean.max(),color='black',s=50,
               edgecolors='purple')
    ax.vlines(mean.argmax(),0,mean.max())
    ax.text(mean.argmax(),mean.max()*1.01,'{:.2f}'.format(mean.max()))
    ax.set_ylim(mean.min()*0.9,mean.max()*1.1)

import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
est = xgb.XGBClassifier