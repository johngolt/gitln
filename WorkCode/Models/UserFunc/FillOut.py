'''
这个模块包括缺失值填充，异常值截断，类别特征编码。
ReadData:包含数据保存，出于后续快速读取的需要保存为pkl.利用pandas_profiling查看数据情况。建立基线模型；
特征变换：box-cox；数据类型转换减少内存消耗。

DistFill:使用分布来对缺失值进行填充，为一个抽象基类。所有的填充方法继承自这个类，必须实现fillmethod
函数，这个函数返回一个可以产生填充值的函数。
DistFillNum:所有对int和float类型的缺失值填充都继承自这个类
DistFillCat:所有对object类型的缺失值填充都继承自这个类
CateFill:填充类别特征的的缺失值填充类
IntRange:对于只能取整数的特征采用Uniform填充
UniformFill:对缺失值采用Uniform填充
GaussianFill:对缺失值采用Gaussian分布填充。

OutlierPro:处理异常值的抽象基类，所有处理异常值的类都是它的子类，所有子类必须实现get_threshold函数。
此函数得到截断的最大和最小值。
BoxTruc：基于箱型图来进行截断，75%+IR,25%-IR
PercentTruc:基于OutlierPro实现的异常值截断的类，最大和最小值为99%和1%

Encode:类别特征进行编码的抽象基类，所有子类必须实现_encode函数，此函数返回一个映射或者其他。
OneHot: one-hot编码
CountEncode：计数编码
LabelCountEncode:计数排序编码
TargetEncode:目标编码
'''


import pandas as pd
import os
import xgboost as xgb
from scipy.stats import boxcox
import pandas_profiling
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
from functools import partial
import re
from sklearn.preprocessing import OneHotEncoder


class ReadData:

    def get_path(self, path):  # 得到文件的目录和文件名
        dirname, name = os.path.split(path)
        name = name.split('.')[0]
        return dirname, name

    def process(self, path, end='.pkl'):
        '''对数据类型进行转换减少内存消耗，同时生产新的文件名，方便后续使用'''
        df = pd.read_csv(path)
        dirname, name = self.get_path(path)
        name = name+end
        df = self.reduce_mem(df)
        return df, dirname, name

    def csv2pkl(self, path):
        '''由于pkl数据格式读取比csv快，可以将格式对数据进行转换后存为pkl格式。'''
        df, dirname, name = self.process(path)
        os.chdir(dirname)
        df.to_pickle(name)

    def csv2hdf(self, path):
        '''hdf读取数据快于csv,在hdf读取的路径中不可以存在中文，否则报错。'''
        df, dirname, name = self.process(path, end='.h5')
        os.chdir(dirname)
        df.to_hdf(name, 'df',mode='w')

    def read_hdf(self, path):
        dirname, name = os.path.split(path)
        os.chdir(dirname)
        df = pd.read_hdf(name)
        return df

    def profiling(self, path):  # 利用pandas_profiling来查看数据情况。
        df, dirname, name = self.process(path, end='.html')
        profile = df.profile_report(title='Profiling Report')
        os.chdir(dirname)
        profile.to_file(output_file=name)
    
    def get_X_y(self, data, target=None):  # 特征和label分离
        data = data.copy()
        y = data.pop(target)
        return data, y
    
    def baseln(self, X, y, scoring='roc_auc', cv=5):
        '''生成基线模型，作为后续数据处理和特征工程的对照。'''
        clf = xgb.XGBClassifier()
        X = pd.get_dummies(X, dummy_na=True)
        res = cross_val_score(clf, X.to_numpy(), y.to_numpy(), scoring=scoring,
                                cv=cv)
        return res.mean()
    
    def bctrans(self, y):  # box-cox转换
        data,lamba = boxcox(y.to_numpy())
        data = pd.Series(data, index=y.index)
        return data, lamba

    def bcinvert(self, y, lamba):  # box-cox逆变换。
        data = np.exp(np.log1p(lamba*y)/lamba)
        data = data.round(2)
        return data
        
    def plot_bc(self, y):  #可视化box-cox
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,2,1)
        sns.distplot(y, hist_kws={'edgecolor':'b'}, ax=ax)
        print('Kurt:{:.2f}, Skew:{:.2f}'.format(y.kurt(), y.skew()))
        data, _ = self.bctrans(y)
        ax1 = fig.add_subplot(1, 2, 2)
        sns.distplot(data, hist_kws={'edgecolor':'b'}, ax=ax1)
        print('After Transform Kurt:{:.2f}, Skew:{:.2f}'.format(data.kurt(), data.skew()))

    def reduce_mem(self, df, verbose=False):
        '''对数据类型进行转化，将对文件内存消耗，所有的int从默认的int64变为int32,所有的float从
        默认的float64变为float32.'''
        dtypes = df.dtypes.astype(str)
        columns = df.columns
        intmask = dtypes.str.contains('int')
        floatmask = dtypes.str.contains('float')
        result = df.copy()
        result.loc[:,columns[intmask]] = result.loc[:,columns[intmask]].astype(np.int32)
        result.loc[:,columns[floatmask]] = result.loc[:,columns[floatmask]].astype(np.float32)
        if verbose:
            start = df.memory_usage().sum()/1024**2
            end = result.memory_usage().sum()/1024**2
            print('Mem. usage is {:5.2f} Mb'.format(end))
            print('reduction rate is {:.1%}'.format((start-end)/start))
        return result


class DistFill:
    '''对缺失值采用分布填充，缺失值填充的抽象基类。'''
    def __init__(self):
        self.mapping_={}

    def float2int(self, ser):
        '''确定缺失值特征的数据类型。和缺失情况的掩码。'''
        mask = pd.notnull(ser)
        if ser.dtype == 'object':
            return 'object', mask
        else:
            temp = ser[mask]
            total = (temp-temp.astype(int)).abs().sum()
            if total < 0.001:
                return np.int32, mask
        return np.float32, mask
    
    @abstractmethod
    def fillmethod(self, ser, **kwargs):
        '''缺失值填补的具体方法，每一个子类实现这个方法。此方法从已有的样本
        产生生成缺失值的方法。'''
        pass

    def fill(self, data, feature, **kwargs):
        ser = data[feature].copy()
        dtype, mask = self.float2int(ser)
        temp = ser[mask]
        fillfunc = self.fillmethod(temp, **kwargs)
        self.mapping_[feature] = fillfunc
        ser[~mask] = fillfunc(size=len(ser[~mask]))
        ser = ser.astype(dtype)
        return ser

    def fills(self, data, features, **kwargs):
        res= []
        for feature in features:
            ser = self.fill(data, feature, **kwargs)
            res.append(ser)
        return pd.concat(res, axis=1)
    
    def transform(self, data, features):
        res = []
        for feature in features:
            ser = data[feature].copy()
            dtype, mask = self.float2int(ser)
            ser[~mask] = self.mapping_[feature](size=len(ser[~mask]))
            ser = ser.astype(dtype)
            res.append(ser)
        return pd.concat(res, axis=1)


class DistFillNum(DistFill):
    '''填充数值特征的父类，用于填充非obejct类型'''
    def get_info(self, ser, robust=False):
        '''得到序列的均值、方差或者中位数和四分位差。'''
        if robust:
            median = np.median(ser)
            std = np.percentile(ser,75)-np.percentile(ser,25)
            return median, std
        mean, std = ser.mean(), ser.std()
        return mean, std
    
    def plot_fill_num(self, data, feature, **kwargs):
        '''可视化填充的缺失值分布和样本未确实的值的分布。'''
        _, mask = self.float2int(data[feature])
        ser = self.fill(data, feature, **kwargs)
        fillv, origin = ser[~mask], ser[mask]
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        sns.distplot(fillv,bins=50,color='r',label='Fill', ax=ax)
        sns.distplot(origin,bins=50, color='g', label='Origin', ax=ax)
        plt.legend()


class DistFillCat(DistFill):
    '''类别特征的父类，用于填充object类型。'''
    def plot_fill_cat(self, data, feature, **kwags):
        _, mask = self.float2int(data[feature])
        ser = self.fill(data, feature, **kwags)
        fillv = ser[~mask].value_counts(normalize=True) 
        origin = ser[mask].value_counts(normalize=True)
        df = pd.concat([origin, fillv],axis=1,keys=['Origin','Fill'],sort=True)
        df = df.fillna(0)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        df.plot(kind='bar',ax=ax)


class CateFill(DistFillCat):
    def fillmethod(self, ser):
        per = ser.value_counts(normalize=True)
        return partial(np.random.choice, a=ser.unique(), p=per)


class IntRange(DistFillNum):
    def fillmethod(self, ser, robust=False):  
        mean, std = self.get_info(ser, robust)
        return partial(np.random.randint, low=mean-std, high=mean+std)


class UniformFill(DistFillNum):
    def fillmethod(self, ser, robust=False):
        mean, std = self.get_info(ser, robust)
        return partial(np.random.uniform, low=mean-std, high=mean+std)


class GaussianFill(DistFillNum):
    def bcinvert(self, loc=0, scale=1, lamba=1, size=1):
        y = np.random.normal(loc, scale, size)
        data = np.exp(np.log1p(lamba*y)/lamba)  # box-cox逆变换
        data = data.round(4)
        return data

    def fillmethod(self, ser, robust=False):
        data, lamba = boxcox(ser)
        mean, std = self.get_info(data, robust)
        return partial(self.bcinvert, loc=mean, scale=std, lamba=lamba)


class OutlierPro:

    @abstractmethod
    def get_threshold(self, series):
        '''给定一个序列，得到盖帽法的最大最小的阈值。由继承的子类实现。'''
        pass

    def truncate(self, data, feature):
        '''执行盖帽发的主函数，返回盖帽法之后的序列和最大最小阈值。'''
        series = data[feature].copy()
        minimum, maximum = self.get_threshold(series)
        upper, lower = max(minimum, maximum), min(minimum, maximum)
        series[series>upper] = upper
        series[series<lower] = lower
        return series, upper, lower
    
    def plot_truncate(self, data, feature):
        '''可视化，通过阈值和和stripplot来可视化阈值的选择，可以根据可视化的结果，在后续的
        处理过程中调整阈值。'''
        series = data[feature].copy()
        minimum, maximum = self.get_threshold(series)
        upper, lower = max(minimum, maximum), min(minimum, maximum)
        mask = (lower<=series)&(series<=upper)
        ax = sns.stripplot(series[mask],orient='v',color='b')
        ax = sns.stripplot(series[~mask], orient='v',color='r')
        ax.axhline(lower,color='y')
        ax.axhline(upper,color='y')
    
    def truncates(self, data, features):
        result = []
        self.mapping_ = { }
        for feature in features:
            series, upper, lower = self.truncate(data, feature)
            result.append(series)
            self.mapping_[features] = (lower, upper)
        return pd.concat(result, axis=1)
    
    def transform(self, data, features):
        '''对于测试集使用，使用训练集得到阈值，然后应用到测试集之上。'''
        result = []
        for feature in features:
            lower, upper = self.mapping_[feature]
            series = data[feature].copy()
            series[series<lower] = lower
            series[series>upper] = upper
            result.append(series)
        return pd.concat(result, axis=1)


class BoxTruc(OutlierPro):  # 具体的异常值处理的类
    def get_threshold(self, series):
        low = series.quantile(0.25)
        high = series.quantile(0.75)
        ir = (high-low) * 1.5
        return low-ir, high+ir


class PercentTruc(OutlierPro):
    def get_threshold(self, series):
        low = series.quantile(0.01)
        high = series.quantile(0.99)
        return low, high


class MixBoxPerTruc(OutlierPro):
    def get_threshold(self, series):
        low = series.quantile(0.25)
        high = series.quantile(0.75)
        ir = (high-low) * 1.5
        low, high = low-ir, high+ir
        low1 = series.quantile(0.01)
        high1 = series.quantile(0.99)
        return min(low, low1), max(high, high1)


class Encode:
    '''对类别特征进行编码，具体的编码方式由子类来实现。对于缺失值需要进行填充之后进行编码，
    本身不能处理缺失值。'''
    def __init__(self):
        self.mapping_ = {}  # 保存编码的映射。

    def fillcat(self, ser):   # 使用None填补类别特征的缺失值。
        df_cat = ser.copy()
        df_cat = df_cat.fillna('None')
        return df_cat

    @abstractmethod
    def _encode(self, data, feature, y=None, **kwargs):
        '''由子类实现具体的编码映射。'''
        pass

    def process(self, data, feature, mapping):
        '''具体实现类别特征的编码，不同的编码方式处理方式可能不同，这个可以根据具体
        的编码方式来决定子类是否重写此方法，如果是一对一的编码，则不需要，如果像onehot那种一对多则
        需要重写此方法。'''
        result = data[feature].map(mapping)
        return result
    
    def encode(self, data, feature, y=None, **kwargs):
        '''实现特征编码的函数。'''
        data = data.copy()
        data[feature] = self.fillcat(data[feature])
        enc = self._encode(data, feature, y, **kwargs)
        result = self.process(data, feature, enc)
        return result

    def fit(self, X, y=None, **kwargs):
        '''对多个特征进行编码，同时为了与sklearn保持一致。'''
        features = X.columns
        result = X.copy()
        for feature in features:
            result[feature] = self.fillcat(result[feature])
            self.mapping_[feature] = self._encode(result, feature, y, **kwargs)
        return self

    def transform(self, X):
        features = X.columns
        result = X.copy()
        res =[]
        for feature in features:
            result[feature] = self.fillcat(result[feature])
            res.append(self.process(result, feature, self.mapping_[feature]))
        res = pd.concat(res, axis=1)
        return res

    def fit_transform(self, X, y=None, **kwargs):
        _ = self.fit(X, y, **kwargs)
        result = self.transform(X)
        return result


class OneHot(Encode):  # OneHotEncode
    def _encode(self, data, feature, y=None):
        enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
        enc.fit(data.loc[:,[feature]])
        return enc

    def process(self, data, feature, enc):
        result = enc.transform(data.loc[:,[feature]]).toarray()
        columns = enc.get_feature_names()
        func = partial(re.sub, 'x0', feature)
        columns = list(map(func, columns))
        result = pd.DataFrame(result, columns=columns, index=data.index)
        return result

class CountEncode(Encode):  # Count-Encode
    def _encode(self, data, feature, y=None, normalize=False):
        temp = data[feature].value_counts() 
        if normalize:
            mapping = temp/temp.max()
        else:
            mapping = temp
        return mapping

class LabelCountEncode(Encode):  # LabelCount-Encode
    def _encode(self, data, feature, y=None, ascending=False):
        mapping = data[feature].value_counts().rank(method='min', 
                                      ascending=ascending)
        return mapping

class TargetEncode(Encode):  # Target-Encode
    def _encode(self, data, feature, y=None):
        temp = pd.concat([data,y],axis=1)
        name = y.name
        mapping = temp.groupby(feature)[name].mean()
        return mapping

class MeanEncode(Encode):  # Mean-Encode
    def _encode(self, data, feature, y=None):
        mean = y.mean()
        name = y.name
        df = pd.concat([data[feature], y],axis=1)
        mapping = df.groupby(feature)[name].mean()
        mapping = mapping + mean
        return mapping