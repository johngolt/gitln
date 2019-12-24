import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder


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



import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class ReadData:
    '''因为pandas读取pickle和hdf的数据更快，所以将csv转换为这两种，
    方便以后继续读取和使用。'''
    def get_path(self, path):
        dirname, name = os.path.split(path)
        name = name.split('.')[0]
        return dirname, name

    def process(self, path, end='.pkl'):
        df = pd.read_csv(path)
        dirname, name = self.get_path(path)
        path1 = os.path.join(dirname, name+end)
        df = self.reduce_mem(df)
        return df, path1

    def csv2pkl(self, path):
        df, path1 = self.process(path)
        df.to_pickle(path1)

    def csv2hdf(self, path):
        df, path1 = self.process(path, end='.hdf')
        df.to_hdf(path1, 'df')

    def read_hdf(self, path):
        df = pd.read_hdf(path,'df')
        return df

    def profiling(self, path):
        df, path1 = self.process(path, end='.html')
        profile = df.profile_report(title='Profiling Report')
        profile.to_file(output_file=path1)
    
    def get_X_y(self, data, target=None):
        y = data[target]
        X = data.pop(target)
        return X, y
    
    def baseln(self, X, y, scoring='roc_auc', cv=5):
        clf = xgb.XGBClassifier()
        X = pd.get_dummies(X, dummy_na=True)
        res = cross_val_score(clf, X.to_numpy(), y.to_numpy(), scoring=scoring,
        cv=cv)
        return res.mean()
    
    def bctrans(self, y):
        data,lamba = boxcox(y.values)
        data = pd.Series(data, index=y.index)
        return data, lamba

    def bcinvert(self, y, lamba):
        data = np.exp(np.log1p(lamba*y)/lamba)
        data = data.round(2)
        return data
    def plot_bc(self, y):
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,2,1)
        sns.distplot(y, hist_kws={'edgecolor':'b'}, ax=ax)
        print('Kurt:{:.2f}, Skew:{:.2f}'.format(y.kurt(), y.skew()))
        data, _ = self.bctrans(y)
        ax1 = fig.add_subplot(1, 2, 2)
        sns.distplot(data, hist_kws={'edgecolor':'b'}, ax=ax1)
        print('After Transform Kurt:{.2f}, Skew:{:.2f}'.format(data.kurt(), data.skew()))

    def reduce_mem(self, df, verbose=False):
        columns = df.columns
        inttype = columns.str.contains('int')
        floatype = columns.str.contains('float')
        result = df.copy()
        result.loc[:,inttype] = result.loc[:,inttype].astype(np.int32)
        result.loc[:,floatype] = result.loc[:,floatype].astype(np.float32)
        if verbose:
            start = df.memory_usage().sum()/1024**2
            end = result.memory_usage().sum()/1024**2
            print('Mem. usage is {:5.2f} Mb'.format(end))
            print('reduction rate is {:.1%}'.format((start-end)/start))
        return result


from abc import abstractmethod
from functools import partial
class DistFill:
    
    def __init__(self):
        self.mapping_={}

    def float2int(self, ser):
        '''确定缺失值特征的数据类型。和缺失值的掩码。'''
        mask = pd.notnull(ser)
        temp = ser[mask]
        total = (temp-temp.astype(int)).abs().sum()
        if total < 0.001:
            return np.int32, mask
        return np.float32, mask

    def get_info(self, ser, robust=False):
        '''得到序列的均值、方差或者中位数和四分位差。'''
        if robust:
            median = ser.median()
            std = np.percentile(ser,75)-np.percentile(ser,25)
            return median, std
        mean, std = ser.mean(), ser.std()
        return mean, std
    
    @abstractmethod
    def fillmethod(self, ser, robust):  # 缺失值填补的方法。
        pass

    def fill(self, data, feature, robust=False):
        ser = data[feature].copy()
        dtype, mask = self.float2int(ser)
        temp = ser[mask]
        fillfunc = self.fillmethod(temp, robust)
        self.mapping_[feature] = fillfunc
        ser[~mask] = fillfunc(size=len(ser[~mask]))
        ser = ser.astype(dtype)
        return ser

    def fills(self, data, features, robust=False):
        res= []
        for feature in features:
            ser = self.fill(data, feature, robust)
            res.append(ser)
        return pd.concat(res, axis=1)
    
    def transform(self, data, features):
        res = []
        for feature in features:
            ser = data[feature].copy
            dtype, mask = self.float2int(ser)
            ser[~mask] = self.mapping_[feature](size=len(ser[~mask]))
            ser = ser.astype(dtype)
            res.append(ser)
        return pd.concat(res, axis=1)


class IntRange(DistFill):
    def fillmethod(self, ser, robust):  
        mean, std = self.get_info(ser, robust)
        return partial(np.random.randint, low=mean-std, high=mean+std)


class UniformFill(DistFill):
    def fillmethod(self, ser, robust):
        mean, std = self.get_info(ser, robust)
        return partial(np.random.uniform, low=mean-std, high=mean+std)


class GaussianFill(DistFill):
    def bcinvert(self, loc=0, scale=1, lamba=1, size=1):
        y = np.random.normal(loc, scale, size)
        data = np.exp(np.log1p(lamba*y)/lamba)
        data = data.round(4)
        return data

    def fillmethod(self, ser, robust=False):
        data, lamba = boxcox(ser)
        mean, std = self.get_info(data, robust)
        return partial(self.bcinvert, loc=mean, scale=std, lamba=lamba)



class OutlierPro:

    @abstractmethod
    def get_threshold(self, series):
        pass

    def truncate(self, data, feature):
        series = data[feature].copy()
        minimum, maximum = self.get_threshold(series)
        upper, lower = max(minimum, maximum), min(minimum, maximum)
        series[series>upper] = upper
        series[series<lower] = lower
        return series, upper, lower
    
    def truncates(self, data, features):
        result = []
        self.mapping_ = { }
        for feature in features:
            series, upper, lower = self.truncate(data, feature)
            result.append(series)
            self.mapping_[features] = (lower, upper)
        return pd.concat(result, axis=1)
    
    def transform(self, data, features):
        result = []
        for feature in features:
            lower, upper = self.mapping_[feature]
            series = data[feature].copy()
            series[series<lower] = lower
            series[series>upper] = upper
            result.append(series)
        return pd.concat(result, axis=1)

class BoxTruc(OutlierPro):
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
