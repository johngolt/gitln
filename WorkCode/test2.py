
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from BayesOpt import BayesianOptimization

def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    estimator = RFC( n_estimators=n_estimators,
        min_samples_split=min_samples_split, max_features=max_features, random_state=2)
        
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,targets=targets,)

    optimizer = BayesianOptimization( f=rfc_crossval,
        pbounds={ "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),},
        random_state=1234,)
    optimizer.maximize(n_iter=10)
    print("Final result:", optimizer.max)


from collections.abc import Iterable
import pandas as pd 
import numpy as np

class CalEnt:
    '''比较某字段在类别合并前后的信息熵、基尼值、信息值IV，若合并后信息熵值减小/基尼值减小/信息值IV增大，
则确认合并类别有助于提高此字段的分类能力，可以考虑合并类别。'''
    def _entropy(self, group):
        return -group*np.log2(group+1e-5)
    
    def _gini(self, group):
        return group*(1-group)
        
    def cal(self, data, feature, target, func=None):
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = func(temp)
        return (temp1*enti.sum(axis=1)).sum()

    def entropy(self, data, feature, target):
        #熵值Ent(D)越小，表示变量的分类能力越强，即预测能力越强。
        if isinstance(feature, str):
            return self.cal(data, feature, target,self._entropy)
        elif isinstance(feature, Iterable):
            return [self.cal(data, each, target, self._entropy) for each in feature]
        else:
            raise TypeError('Feature is not right data type')

    def gini(self, data, feature, target):
        # 基尼值越小，说明数据集的纯度越高，即变量的预测力越强。
        if isinstance(feature,str):
            return self.cal(data, feature, target, self._gini)
        elif isinstance(feature,Iterable):
            return [self.cal(data, each, target, self._gini) for each in feature]
        else:
            raise TypeError('Feature is not right data type')

    def Iv(self, data, feature, target):
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = np.log((temp.iloc[:,0]+1e-5)/(temp.iloc[:,1]+1e-5))
        iv = (temp.iloc[:,0] - temp.iloc[:,1])*woei
        return iv.sum(), woei.to_dict()

    def woe_iv(self,data, feature, target):
        if isinstance(feature,str):
            return self.Iv(data,feature, target)
        elif isinstance(feature,Iterable):
            return{each:self.Iv(data, each, target) for each in feature}
        else:
            raise TypeError('Feature is not right data type')