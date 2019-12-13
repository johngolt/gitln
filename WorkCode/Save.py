
import xgboost as xgb
import lightgbm as lgb
from abc import abstractmethod
from sklearn.model_selection import GridSearchCV,StratifiedKFold, KFold
from itertools import repeat
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import clone
import numpy as np 
from sklearn.metrics import roc_auc_score

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


        
    