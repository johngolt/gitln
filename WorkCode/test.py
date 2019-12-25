import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
class Encode:
    def __init__(self):
        self.mapping_ = {}

    @abstractmethod
    def _encode(self, data, feature, y, **kwargs):
        pass

    def process(self, data, feature, enc):
        if callable(enc):
            result = enc(data[feature])
        else:
            result = data[feature].map(enc)
        return result
    
    def encode(self, data, feature, y=None, **kwargs):
        enc = self._encode(data, feature, y, **kwargs)
        result = self.process(self, data, feature, enc)
        return result

    def fit(self, X, y, **kwargs):
        features = X.columns
        result = X.copy()
        for feature in features:
            self.mapping_[feature] = self._encode(result, feature, y, **kwargs)
        return self

    def transform(self, X, **kwargs):
        features = X.columns
        result = X.copy()
        for feature in features:
            result[feature] = self.process(result, feature, self.mapping_[feature])
        return result

    def fit_transform(self, X, y, **kwargs):
        _ = self.fit(X, y, **kwargs)
        result = self.transform(self, X, **kwargs)
        return result

class OneHot(Encode):
    def _encode(self, data, feature, y):
        enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
        return enc.fit



class Encoding: 
    '''类别特征编码：one-hot, Count, LabelCount, 目标编码'''
    def __init__(self):
        self.count_encode_map = { }
        self.labelcount_encode_map={ }
        self.target_encode_map = { }
        self.onehot_encode_map = { }
        
    def onehot_encode(self, data, features):
        import re
        pattern = re.compile(r'x(\d+)')
        enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
        enc.fit(data[features])
        self.onehot_encode_map[tuple(features)] = enc
        temp = enc.transform(data[features]).toarray()
        self.columns = [features[int(pattern.search(item).groups(1)[0])]+'_'+item.split('_') [-1]
                   for item in enc.get_feature_names()]
        df = pd.DataFrame(temp,index = data.index,columns=self.columns) 
        return df
    
    def count_encode(self, data, features, normalize=False):
        df = pd.DataFrame()
        for feature in features:
            temp = data[feature].value_counts() 
            if normalize:
                mapping = temp/temp.max()
            else:
                mapping = temp
            self.count_encode_map[feature] = mapping
            df[feature] = data[feature].astype('object').map(mapping)
        if normalize:
            df = df.astype(np.float32)
        else:
            df = df.astype(np.float32)
        df = df.add_suffix('_count_encode')
        return df
    
    def labelcount_encode(self, data, features, ascending=False):
        df = pd.DataFrame()
        for feature in features:
            mapping = data[feature].value_counts(
            ).rank(method='min', ascending=ascending)
            self.labelcount_encode_map[feature] = mapping
            df[feature] = data[feature].astype('object').map(mapping)
        df = df.astype(np.float32)
        df = df.add_suffix('_labelcount_encode')
        return df
    
    def target_encode(self, data, features, y):
        name = y.name
        df = pd.DataFrame()
        temp = pd.concat([data,y],axis=1)
        for feature in features:
            mapping = temp.groupby(feature)[name].mean()
            self.target_encode_map[feature] = mapping
            df[feature] = data[feature].astype('object').map(mapping)
        df = df.add_suffix('_target_encode')
        df = df.astype(np.float32)
        return df
    
    def transform(self, Xtest):
        df = pd.DataFrame()
        for key in self.count_encode_map:
            df[key+'_count_encode'] = Xtest[key].map(
                self.count_encode_map[key])
        for key in self.labelcount_encode_map:
            df[key+'_labelcount_encode'] = Xtest[key].map(
            self.labelcount_encode_map[key])
        for key in self.target_encode_map:
            df[key+'_target_encode'] = Xtest[key].map(
            self.target_encode_map[key])
        if self.onehot_encode_map != {}:
            features, enc = list(self.onehot_encode_map.items())[0]
            temp = enc.transform(Xtest[list(features)]).toarray()
            df = pd.DataFrame(temp,index=Xtest.index,columns=self.columns)
        df.astype(np.float32)
        return df


class TargetEncode:
    def __init__(self):
        self.map_ = {}
        self.mean_ = None
        self.none_ = {}
        
    def fit(self,X,y):
        self.mean_ = y.mean()
        name = y.name
        df = pd.concat([X,y],axis=1)
        for each in X:
            data = df.groupby(each)[name].mean()
            self.map_[each] = data.to_dict()
            self.none_[each] = y[pd.isnull(df[each])].mean()
        return self
    
    def transform(self,X):
        result = X.copy()
        for each in X:
            result[each] = X[each].map(
                lambda x,column=each: self.func(x,column) + self.mean_)
        return result
    
    def func(self,x,column=None):
        if pd.isnull(x):
            return self.none_.get(column,0)
        else:
            return self.map_[column].get(x,0)



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
