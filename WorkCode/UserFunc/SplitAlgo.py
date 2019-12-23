import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from abc import abstractmethod
from functools import partial

class Split:

    def __init__(self, bins=10):  # 设置默认的分箱数
        self.bins = bins

    def get_bins(self, bins=None):  # 获取分箱数
        if bins is None:
            return self.bins
        return bins

    def get_series(self, data, feature=None, target=None):  # 处理Series和DataFrame
        if feature is None:
            return data
        if target is None:
            return data[feature]
        else:
            return data[[feature, target]]
    
    @abstractmethod
    def _split(self, ser, bins, feature=None, target=None):  # 分箱函数，由子类实现。
        pass
    
    def split(self, data, feature, target=None, bins=None):  # 对序列进行分箱。有监督和无监督。
        ser = self.get_series(data, feature, target)
        bins = self.get_bins(bins)
        res, points = self._split(ser, bins, feature=feature, target=target)
        points[0], points[-1] = -np.inf, np.inf
        mapping = {key:value for value,key in enumerate(res.cat.categories)}
        res = res.cat.rename_categories(mapping)
        return res, points, mapping
    
    def splits(self, data, features, target=None, bins=None):  # 对多个连续特征进行分箱。有监督和无监督
        bins = self.get_bins(bins)
        self.splitpoints = {}
        self.mappings = {}
        if isinstance(features, str):
            res, points, mapping = self.split(data, features, bins)
            self.splitpoints[features] = points
            self.mappings[features] = mapping
            return res
        result = []
        for feature in features:
            res, points, mapping = self.split(data, feature, bins)
            self.splitpoints[feature] = points
            self.mappings[features] = mapping
            result.append(res)
        result = pd.concat(result, axis=1)
        return result

class FreqSplit(Split):
    def _split(self, ser, bins):
        res, points = pd.cut(ser, bins=bins, retbins=True)
        return res, points

class LengthSplit(Split):
    def _split(self, ser, bins):
        res, points = pd.qcut(ser, q=bins, duplicates='drop', retbins=True)
        return res, points


class TreeSplit(Split):

    def __init__(self, bins=10):
        self.bins=bins

    @abstractmethod
    def _entropy(self, group):
        pass

    def cal(self, data, feature, target):
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = self._entropy(temp)
        return (temp1*enti.sum(axis=1)).sum()

    def increase(self, data, feature, target):
        end = self.cal(data, feature, target)
        tmp = data[target].value_counts(normalize=True)
        begin = self._entropy(tmp).sum()
        return begin - end
    
    def bestpoint(self, data, feature, target):
        df = data.loc[:,[feature, target]].copy()
        values = data[feature].unique()
        if len(values) == 1:
            return pd.Series([values, 0],index=['value','increase'])
        values.sort()
        values = (values[:-1]+values[1:])/2
        dic = {}
        for value in values:
            mask = data[feature] < value
            df.loc[mask, feature] = 0
            df.loc[~mask, feature] = 1
            dic[value] = self.increase(df, feature, target)
        result = sorted(dic.items(), key=lambda x:x[1],reverse=True)[0] 
        result = pd.Series(result,index=['value','increase'])
        return result  #返回value和increase.
    
    def generalsplit(self, data, feature, target):
        values = [-np.inf, np.inf]
        baseln = 0.001
        while (len(values) <= self.bins+1) and (baseln >= 0.001):
            values.sort()
            df = data[[feature, target]].copy()
            grouper = pd.cut(df[feature], bins=values)
            func = partial(self.bestpoint, feature=feature, target=target)
            res = df.groupby(grouper).apply(func)
            index = res['increase'].idxmax()
            values.append(res.loc[index,'value'])
            baseln = res.loc[index,'increase']
        return values

    def _split(self, ser, bins, feature=None, target=None):
        values = self.generalsplit(ser, feature, target)
        values.sort()
        res = pd.cut(ser[feature], bins=values)
        return res, values

class EntropySplit(TreeSplit):
    def _entropy(self, group):
        return -group*np.log2(group+1e-5)


class GiniSplit(TreeSplit):
    def _entropy(self, group):
        return group*(1-group)


class Merge(Split):
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    @abstractmethod
    def diff(self, data, index1, index2, feature, target):
        pass

    def dfmerge(self, data, index1, index2, feature, target):
        df1 = data.loc[index1,[feature, target]]
        df1[feature] = 0
        df2 = data.loc[index2, [feature, target]]
        df2[feature] = 1
        df = pd.concat([df1,df2])
        return df

    def chergepoint(self, data, feature, target):
        values = data[feature].unique()
        values.sort()
        values = (values[:-1] + values[1:])/2
        values = values.tolist()
        baseln = self.threshold
        while (len(values) >self.bins+1) and (baseln <= self.threshold):
            df = data[[feature, target]].copy()
            grouper = pd.cut(data[feature], bins=values)
            group = df.groupby(grouper).groups
            keys = list(group.keys())[1:]
            dvalues = list(group.values())[:-1]
            remove = 0
            for value, key, dvalue in zip(values[1:],keys, dvalues):
                res = self.diff(df, dvalue, group[key], feature, target)
                if res <= baseln:
                    baseln = res
                    remove = value
            value.remove(remove)
        return values

    def _split(self, ser, bins, feature=None, target=None):
        values = self.chergepoint(ser, feature, target)
        values.sort()
        res = pd.cut(ser[feature], bins=values)
        return res, values
            

class WOESplit(Merge):
    def diff(self, data, index1, index2, feature, target):
        df = self.dfmerge(data, index1, index2, feature, target)
        temp = pd.crosstab(df[feature], df[target], normalize='columns')
        woe = np.log((temp.iloc[:, 0]+1e-5)/(temp.iloc[:, 1]+1e-5))
        return woe[0] - woe[1]


class ChiMerge(Merge):
    def diff(self, data, index1, index2, feature, target): 
        df = self.dfmerge(data, index1, index2, feature, target)
        arr = pd.crosstab(df[feature], df[target])
        a = arr.sum(axis=0)
        b = arr.sum(axis=1)
        demonitor = a.prod()*b.prod()
        nomitor = (arr[0,0]*arr[1,1]-arr[0,1]*arr[1,0])**2*arr.sum()
        return nomitor/demonitor






    



    

