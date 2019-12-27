
import xgboost as xgb
import lightgbm as lgb
from abc import abstractmethod
from sklearn.model_selection import GridSearchCV,StratifiedKFold, KFold
from itertools import repeat
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import clone
import numpy as np 
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV, GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import minepy
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

'''Constructs a new estimator with the same parameters.
Clone does a deep copy of the model in an estimator
without actually copying attached data. It yields a new estimator
with the same parameters that has not been fit on any data.'''

class SelectHyparams:
    def __init__(self, estimator, params, imbalance=False):
        self.estimator = estimator.set_params(**params)
        self.imbalance = imbalance
        self.params = params
        self.scoring = 'roc_auc'

    def set_scoring(self, scoring): # 设置评价函数
        self.scoring = scoring

    def set_params(self, params): # 设置参数
        self.params = {**self.params, **params}
        self.estimator = self.estimator.set_params(**params)

    def set_param_grid(self, param_grid): # 设置调参的值
        if hasattr(self, 'param_grid'):
            self.param_grid = {**self.param_grid, **param_grid}
        else:
            self.param_grid = param_grid
            
    def get_cv(self): # 得到cross validation
        if self.imbalance: 
            cv= StratifiedKFold(5,shuffle=True,random_state=10)
        else: 
            cv = KFold(5, shuffle=True, random_state=10)
        return cv

    def _search(self, X, y, param_grid): # GridSearch
        estimator = clone(self.estimator)
        cv = self.get_cv()
        
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                scoring=self.scoring, cv=cv, verbose=0, n_jobs=-1)
        grid.fit(X, y)
        self.params = grid.best_params_
        self.param_grid = param_grid
        estimator = clone(self.estimator)
        estimator.set_params(**self.params)
        self.score = grid.best_score_
        return estimator
    
    def search(self, X, y, param_grid=None):
        estimator = clone(self.estimator)

        if param_grid is None:
            if hasattr(self, 'param_grid'):
                return self._search(X, y, self.param_grid)
            else:
                return estimator
        else:
            return self._search(X, y, param_grid)

    
    def fit(self, X, y, param_grid=None):
        estimator = self.search(X,y, param_grid)
        self.estimator = estimator.fit(X,y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class GradientTree(SelectHyparams):
        
    def get_params(self, params=None):
        if params is None:
            params = self.params
        else:
            params = params
        if 'n_estimators' in params:
            params.pop('n_estimators')
        return params
        
    @abstractmethod
    def get_estimators(self, X, y, params=None):
        pass

    def _search(self, X, y, param_grid):
        estimator = clone(self.estimator)
        self.params['n_estimators']= self.get_estimators(X,y)
        estimator.set_params(**self.params)
        cv = self.get_cv()
            
        score = []
        for key,value in param_grid.items():
            grid = GridSearchCV(estimator=estimator, param_grid={key:value}, 
                        scoring=self.scoring, cv=cv, verbose=0, n_jobs=-1)
            grid.fit(X, y)
            self.params[key] = grid.best_params_[key]
            score.append(grid.best_score_)
            estimator = clone(self.estimator)
            estimator.set_params(**self.params)
        self.score = score
        
        estimator = clone(self.estimator)
        self.params['n_estimators'] = self.get_estimators(X,y)
        estimator.set_params(**self.params)
        return estimator
        
    
    def fit(self, X, y, param_grid=None):
        if param_grid is None:
            self.estimator = self.estimator.fit(X,y)
        else:
            estimator = self.search(X,y, param_grid)
            self.estimator = estimator.fit(X,y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    
class XGB(GradientTree):
    def __init__(self, imbalance=False):
        self.params = { 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1,
                        'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0,
                        'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 5,
                        'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0,
                        'reg_lambda': 0, 'scale_pos_weight': 1,'subsample': 0.5 }

        self.param_grid ={'max_depth':range(1,8,1),
                                  'min_child_weight':range(1,10,1),# 需要精调
                                  'max_leaf_nodes':range(5,31,5),
                                  'gamma':[i/10.0 for i in range(0,5)],
                                  'subsample':[i/10.0 for i in range(5,10)],
                                  'colsample_bytree':[i/10.0 for i in range(5,10)],
                                  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                                  'reg_lambda':[0, 0.001, 0.005, 0.01, 0.05]}
        self.estimator = xgb.XGBClassifier()
        self.imbalance = imbalance
        

    def get_estimators(self, X, y, params=None):
        params = self.get_params(params)
        params['eval_metric'] = 'auc'
        trainset = xgb.DMatrix(X,label=y)
        result = xgb.cv(params, trainset, num_boost_round=500,
               nfold=5,stratified=self.imbalance,early_stopping_rounds=20)
        n_estimators = result['test-auc-mean'].to_numpy().argmax()
        return n_estimators

    def fit(self,X,y):
        estimator = self.search(X,y, self.param_grid)
        self.estimator = estimator.fit(X,y)
        return self
    
class LGBM(GradientTree):
    def __init__(self, imbalance=False):
        self.params = { 'boosting_type': 'gbdt','class_weight': None,'colsample_bytree': 0.5,
                        'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': 5,
                        'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
                        'n_estimators': 200, 'n_jobs': -1, 'num_leaves': 10, 'reg_alpha': 0.0,
                        'reg_lambda': 0.0, 'subsample': 0.5, 'subsample_for_bin': 200000,
                        'subsample_freq': 5}
        
        self.param_grid={'max_depth': range(3,8,1),
                         'num_leaves':range(5, 31, 1),
                         'min_child_samples': range(20,121,20),
                         'min_child_weight':[0.001, 0.005, 0.01, 0.02, 0.03],
                         'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                         'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                         'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
                         'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
                         'min_split_gain':[0, 0.001, 0.01, 0.03, 0.05,0.08]}
        
        self.estimator = lgb.LGBMClassifier()
        self.imbalance = imbalance
        
    def get_estimators(self, X, y, params=None):
        params = self.get_params(params)
        params['metric'] = 'auc'
        trainset = lgb.Dataset(X, label=y)
        result = lgb.cv(params, trainset, num_boost_round=500, nfold=5,
                       stratified=self.imbalance, early_stopping_rounds=20)
        n_estimators = np.array(result['auc-mean']).argmax()
        return n_estimators
    
    def fit(self,X,y):
        estimator = self.search(X,y, self.param_grid)
        self.estimator = estimator.fit(X,y)
        return self

class GBDT(GradientTree):
    def __init__(self, imbalance=False):

        self.params = {'learning_rate': 0.1,'max_depth': 5,
                                'max_features': 0.5, 'max_leaf_nodes': 10,
                                'min_impurity_decrease': 0.0,'min_samples_leaf': 30,
                                'min_samples_split': 50, 'min_weight_fraction_leaf': 0.0,
                                'n_estimators': 100, 'subsample': 0.5,}

        self.param_grid = {'max_depth':range(1,8,1), 
                           'min_samples_split':range(50,201,10),
                          'min_samples_leaf':range(30,101,10),
                          'max_features':np.arange(0.5,1.01,0.1),
                          'subsample':range(0.5,1.01,0.1),'max_leaf_nodes':np.range(5,31,5),}
        self.estimator = GradientBoostingClassifier()
        self.imbalance = imbalance

    def get_estimators(self, X, y, params=None):
        params = self.get_params(params)
        if self.imbalance:
            cv = StratifiedKFold(5,shuffle=True, random_state=10)
        else:
            cv = KFold(5, shuffle=True, random_state=10)

        result = []
        for tid, vid in cv.split(X, y):
            trainx, validx = X.loc[tid,:], X.loc[vid,:]
            trainy, validy = y[tid], y[vid]
            estimator = clone(self.estimator).set_params(**params)
            estimator.fit(trainx, trainy)
            result.append(list(map(lambda y,ypred: roc_auc_score(y,ypred[:,1]), 
                 repeat(validy), estimator.staged_predict_proba(validx))))
        n_estimators = np.array(result).mean(axis=0).argmax()
        return n_estimators

    def fit(self,X,y):
        estimator = self.search(X,y, self.param_grid)
        self.estimator = estimator.fit(X,y)
        return self




class FeatureSelection:
    def __init__(self,estimator=None, to_select=None, score='roc_auc'):
        if estimator is None:  # 用于进行wrapper方法的模型
            self.estimator = RandomForestClassifier(n_estimators=10, max_depth=4)
        else:
            self.estimator = estimator
        self.score = score  # 评估标准
        self.masked = None  # 得到的mask掩码，用于测试集
        
    def select_by_filter(self, func, X, y):
        # filter方法进行可视化。
        to_select = X.shape[1]//2
        '''Function taking two arrays X and y, and returning a pair of arrays  
    (scores, pvalues). For modes 'percentile' or 'kbest' it can return  
    a single array scores. '''
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
        
    def __call__(self, X, y=None, func=None):
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



def decorate(f):  # 装饰器，使得可以使用DataFrame。
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
    
    def plot_corr(self, data):  # 相关系数的可视化
        _, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.set_ylabel('Correlogram')
        dim = data.values.ndim
        assert dim in (1,2)
        if dim == 2:  # heatmap的可视化
            sns.heatmap(data, xticklabels=data.columns, yticklabels=data.columns, cmap='RdYlGn',
            center=0, annot=True,ax=ax)
        else: 
            data=data.sort_values().reset_index()  # 特征与label之间的相关性的可视化。
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


        
    