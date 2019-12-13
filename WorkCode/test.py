import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

class Encoding:
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
        self.columns = [features[int(pattern.search(item
                        ).groups(1)[0])]+'_'+item.split('_') [-1]
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


class ProcessData:
    def __init__(self,cat_fet=None, num_fet=None):
        self.enc = None
        self.cat_fet = cat_fet
        self.num_fet = num_fet
    
    def fit(self, X, y=None):
        result = X.copy()
        cat = result[self.cat_fet]
        enc = TargetEncode()
        self.enc = enc.fit(cat, result['Label'])
        return self 

    def transform(self, X):
        result = X.copy()
        num,cat = result[self.num_fet], result[self.cat_fet]
        result['mean'] = num.mean(axis=1)
        result['std'] = num.std(axis=1)
        result['skew'] = num.skew(axis=1)
        result['median'] = num.median(axis=1)
        result[self.cat_fet] = self.enc.transform(cat)
        return result


def processTrainTest(cmiss1, test2, estimator1, fs=True, valid=False):
    pcmiss1 = ProcessData()
    pcmiss1.fit(cmiss1)
    presult1 = pcmiss1.transform(cmiss1)
    ptest1 = pcmiss1.transform(test2)

    presult1 = presult1.fillna(0)
    ptest1 = ptest1.fillna(0)

    pX1 = presult1.drop(['Label'],axis=1)
    py1 = presult1['Label']
    pX1 = pX1.drop(['ID','投资总额'],axis=1)
    pXt1 = ptest1.drop(['ID','投资总额'],axis=1)
    if valid:
        pXt1 = pXt1.drop(['Label'], axis=1)

    es1 = estimator1.fit(pX1.values,py1)
    if fs:
        dic ={key:value for key, value in zip(pX1.columns,
                            es1.feature_importances_) if value != 0}
        columns = list(dic.keys())
        es1 = estimator1.fit(pX1[columns].values, py1)
        pred1 = es1.predict_proba(pXt1[columns].values)[:,1]
    else:
        pred1 = es1.predict_proba(pXt1.values)[:,1]
    return pred1, ptest1, presult1

