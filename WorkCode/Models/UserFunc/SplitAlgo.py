'''
此模块主要用于对数值特征进行分箱。
Split:是所有分箱函数的抽象基类，所有分箱方法都基于这个类。实现了分箱的通用方法。
LegthSplit:等宽分箱方法。
FreqSplit:等频分箱方法。

TreeSplit：是所有基于分裂式分箱的方法的抽象基类。
GiniSplit:基于基尼系数的分箱方法。
EntropySplit:基于信息熵的分箱方法。
BestKS:基于ks值的分箱方法。

Merge:是所有合并式分箱方法的抽象基类。
WOESplit:基于WOE值的分箱方法。
ChiMerge:基于卡方值的分箱方法。
'''


import pandas as pd
import numpy as np
from abc import abstractmethod
from functools import partial


class Split:

    def __init__(self, bins=10):
        '''设定分箱数，默认为10，可以根据具体情况设定分箱数。'''
        self.bins = bins

    def get_bins(self, bins=None):
        '''获得分箱数，默认为初始化设定的分析数，也可以根据具体的分箱算法来选择。'''
        if bins is None:
            return self.bins
        return bins

    def get_series(self, data, feature=None, target=None):
        '''可以处理DataFrame，也可以直接输入Series'''
        if feature is None:
            return data
        if target is None:
            return data[feature]
        else:
            return data[[feature, target]]

    @abstractmethod
    def _split(self, ser, bins, feature=None, target=None, **kwargs):
        '''分箱的实现函数，返回分箱的结果和分箱点。'''
        pass

    def split(self, data, feature=None, target=None, bins=None, **kwargs):
        '''对序列进行分箱。是一个通用的结构，具体的切分方法在_split中实现，
        这里是对切分之后进行处理，包括保存切分点和对每个区间进行自然数编码。'''
        ser = self.get_series(data, feature, target)
        bins = self.get_bins(bins)

        res, points = self._split(ser, bins, feature=feature, target=target, **kwargs)

        points[0], points[-1] = -np.inf, np.inf
        mapping = {key: value for value, key in enumerate(res.cat.categories)}
        res = res.cat.rename_categories(mapping)
        return res, points, mapping

    def fit(self, X, y=None, bins=None, **kwargs):
        bins = self.get_bins(bins)
        self.splitpoints = {}
        self.mappings = {}
        features = X.columns
        target = y.name
        data = pd.concat([X,y],axis=1)
        for feature in features:
            _, points, mapping = self.split(data, feature, target, bins, **kwargs)
            self.splitpoints[feature] = points
            self.mappings[feature] = mapping
        return self
    
    def transform(self, X):
        result = []
        features = X.columns
        for feature in features:
            res = pd.cut(X[feature], bins=self.splitpoints[feature])
            res = res.cat.rename_categories(self.mappings[feature])
            result.append(res)
        return pd.concat(result, axis=1)

    def splits(self, data, features, target=None, bins=None, **kwargs):
        '''对单或多个连续特征进行分箱。整个分箱算法的对外接口。'''
        bins = self.get_bins(bins)
        self.splitpoints = {}
        self.mappings = {}
        if isinstance(features, str):
            res, points, mapping = self.split(data, features, target, bins, **kwargs)
            self.splitpoints[features] = points
            self.mappings[features] = mapping
            return res
        result = []
        for feature in features:
            res, points, mapping = self.split(data, feature, target, bins, **kwargs)
            self.splitpoints[feature] = points
            self.mappings[features] = mapping
            result.append(res)
        result = pd.concat(result, axis=1)
        return result
        

class LengthSplit(Split):
    def _split(self, ser, bins, feature=None, target=None):
        res, points = pd.cut(ser, bins=bins, retbins=True)
        return res, points


class FreqSplit(Split):
    def _split(self, ser, bins, feature=None, target=None):
        res, points = pd.qcut(ser, q=bins, duplicates='drop', retbins=True)
        return res, points


class TreeSplit(Split):
    '''基于树的分箱方法，包括了基尼系数和信息熵。'''
    
    def __init__(self, thres=0.001, bins=10):
        self.thres = thres
        super().__init__(bins=bins)

    @abstractmethod
    def _entropy(self, group):
        '''计算信息熵和基尼系数的函数。'''
        pass

    def cal(self, data, feature, target):
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = self._entropy(temp)
        return (temp1*enti.sum(axis=1)).sum()

    def increase(self, data, feature, target):
        '''计算增益率。'''
        end = self.cal(data, feature, target)
        tmp = data[target].value_counts(normalize=True)
        begin = self._entropy(tmp).sum()
        return begin - end

    def bestpoint(self, data, feature, target):
        '''寻找连续值的最优切分点。返回最优切分点和对应的信息增益，对特征的取值从小到大遍历，
        分成二叉树。'''
        data = data.copy()
        df = data.loc[:, [feature, target]].copy()
        values = data[feature].unique()
        if len(values) == 1:  # 如果为一个值，无法分箱，返回。
            return pd.Series([values, 0], index=['value', 'increase'])
        values.sort()
        values = (values[:-1]+values[1:])/2
        dic = {}

        for value in values:
            mask = data[feature] < value # 修改了df的值，但并没有修改data中feature的值
            df.loc[mask, feature] = 0
            df.loc[~mask, feature] = 1
            dic[value] = self.increase(df, feature, target)
        result = sorted(dic.items(), key=lambda x: x[1], reverse=True)[0]
        result = pd.Series(result, index=['value', 'increase'])
        return result  # 返回value和increase.

    def generalsplit(self, data, feature, target):
        '''切分过程中，每次增加一个分箱，从所有分箱中找到一个分箱的切点使得
        信息增益最大的那个作为新的分箱点，并添加到values中，在得到n个分箱的情况下，通过对数组进行分箱，
        然后从各个分箱中找到最优的切分点，然后选取信息增益最大的分箱的切分点作为下一个切分点。'''
        values = [-np.inf, np.inf]
        baseln = self.thres

        while (len(values) <= self.bins+1) and (baseln >= self.thres):
            values.sort()

            df = data[[feature, target]].copy() 
            grouper = pd.cut(df[feature], bins=values)
            func = partial(self.bestpoint, feature=feature, target=target)
            res = df.groupby(grouper).apply(func)

            index = res['increase'].idxmax()
            values.append(res.loc[index, 'value'])
            baseln = res.loc[index, 'increase']  # 当上一轮的最大信息增益大于设定值时才会继续分箱。
        return values

    def _split(self, ser, bins, feature=None, target=None):
        values = self.generalsplit(ser, feature, target)
        values.sort()
        res = pd.cut(ser[feature], bins=values)
        return res, values


class BestKS(TreeSplit):
    def increase(self, data, feature, target):
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        temp = temp.cumsum()
        return np.abs(temp.iloc[:,0]-temp.iloc[:,1]).max()


class EntropySplit(TreeSplit):
    def _entropy(self, group):
        return -group*np.log2(group+1e-5)


class GiniSplit(TreeSplit):
    def _entropy(self, group):
        return group*(1-group)


class Merge(Split):
    def __init__(self, threshold=0.01, bins=10):
        self.threshold = threshold  # 合并的最小阈值，当大于这个值是，不再进行合并。
        super().__init__(bins=bins)

    @abstractmethod
    def diff(self, data, index1, index2, feature, target):
        '''计算相邻两个分箱之间的差异，由子类实现。'''
        pass

    def dfmerge(self, data, index1, index2, feature, target):
        '''将两个分箱合并为一个，并标识出来。'''
        df1 = data.loc[index1, [feature, target]].copy()
        df1[feature] = 0
        df2 = data.loc[index2, [feature, target]].copy()
        df2[feature] = 1
        df = pd.concat([df1, df2])
        return df

    def chergepoint(self, data, feature, target):
        values = data[feature].unique()
        values.sort()
        values = (values[:-1] + values[1:])/2
        values = values.tolist()
        values = [-np.inf] + values+[np.inf]
        baseln = self.threshold
        while (len(values) > self.bins+1) and (baseln <= self.threshold):
            df = data[[feature, target]].copy()
            grouper = pd.cut(data[feature], bins=values)
            group = df.groupby(grouper).groups #返回字典
            keys = list(group.keys())[1:]  # 分组的keys
            dvalues = list(group.values())[:-1] # 每个分箱的index
            remove = None
            for value, key, dvalue in zip(values[1:], keys, dvalues):
                res = self.diff(df, dvalue, group[key], feature, target)

                if res <= baseln: # 合并差异小于设定值的区间
                    baseln = max(self.threshold, res)
                    remove = value
                    values.remove(remove)
            if remove is None:
                break
        return values

    def _split(self, ser, bins, feature=None, target=None):
        values = self.chergepoint(ser, feature, target)
        values.sort()
        res = pd.cut(ser[feature], bins=values)
        return res, values


class WOESplit(Merge):
    def diff(self, data, index1, index2, feature, target):
        '''计算相邻分箱之间的woe的差异。'''
        df = self.dfmerge(data, index1, index2, feature, target)
        temp = pd.crosstab(df[feature], df[target], normalize='columns')
        if (0 not in temp.columns) or (1 not in temp.columns) or (0 in temp.to_numpy()):
            return self.threshold
        woe = np.log((temp.iloc[:, 0])/(temp.iloc[:, 1]))
        return woe[0] - woe[1]


class ChiMerge(Merge):
    def diff(self, data, index1, index2, feature, target):
        '''计算相邻分箱的卡方值'''
        df = self.dfmerge(data, index1, index2, feature, target)
        temp = pd.crosstab(df[feature], df[target],normalize='index')
        if (0 not in temp.columns) or (1 not in temp.columns) or (0 in temp.to_numpy()):
            return self.threshold
        arr = temp.to_numpy()
        a = arr.sum(axis=0)
        b = arr.sum(axis=1)
        demonitor = a.prod()*b.prod()
        nomitor = (arr[0, 0]*arr[1, 1]-arr[0, 1]*arr[1, 0])**2*arr.sum()
        return nomitor/demonitor

    
