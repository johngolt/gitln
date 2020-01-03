import numpy as np 
import pandas as pd
from abc import abstractmethod
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

class ResultShow:

    def process(self, df):
        headers = df.columns
        rows = df.to_numpy().tolist()
        return headers, rows
        
    def table(self, df, title=None):
        tb = Table()
        headers, rows = self.process(df)
        tb.add(headers, rows).set_global_opts(
        title_opts=ComponentTitleOpts(title=title))
        return tb

    def get_group(self, y, ypred):
        category, bins = pd.qcut(ypred, q=10, duplicates='drop', retbins=True)
        cate = category.set_categories(bins[:-1], rename=True)
        return cate, bins[:-1][::-1]

    def bin_info(self, y, ypred):
        columns=['箱','分数','样本数量','关注样本数','区间关注类别比例',
        '累计关注类别比例','区间准确率','累计准确率','尾部准确率',
        '提升','累计提升']
        cate, bins = self.get_group(y, ypred)
        df = pd.DataFrame(columns=columns, index=bins)
        group = y.groupby(cate)
        df['箱'] = range(1, len(df)+1)
        df['分数'] = tb.index
        df['样本数量'] = group.size()
        df['关注样本数'] = group.sum()
        df['区间关注类别比例'] = df['关注样本数']/y.sum()
        df['累计关注类别比例'] = df['区间关注类别比例'].cumsum()
        df['区间准确率'] = 
        df['提升'] = 
        df['累计提升'] = 

    def confusin(self, y, ypred, bin_num=1):
        columns=['实际类别','类别总数','正确百分比','预测类别 非关注','预测类别 关注']
        indexs=['非关注','关注','合计']
        df = pd.DataFrame(columns=columns)
        df['实际类别'] = indexs
        df = df.set_index('实际类别')
        df['类别总数'] = [len(y)-y.sum(), y.sum(), len(y)]

        _, bins = self.get_group(y, ypred)
        y1 = (ypred > bins[bin_num-1]).astype(int)

        df.iloc[:2, -2:] = metrics.confusion_matrix(y, y1)
        df.loc['关注','正确百分比'] = df.at['关注','预测类别 关注']/df.at['关注','类别总数']
        df.loc['非关注','正确百分比'] = df.at['非关注','预测类别 非关注']/df.at['非关注','类别总数']
        df.loc['合计','预测类别 关注'] = df.at['关注','预测类别 关注']/df.loc[:,'预测类别 关注'].sum()
        df.loc['合计','预测类别 非关注'] = df.at['非关注','预测类别 非关注']/df.loc[:,'预测类别 非关注'].sum()
        return df

    def indicate(self, y, ypred, bin_num=1):
        _, bins = self.get_group(y, ypred)
        y1 = (ypred > bins[bin_num-1]).astype(int)
        columns=['名称','正确百分比']
        indexs = ['准确率','精确率','召回率']
        indexs2 = ['灵敏度','特异度','F1统计']
        df1 = pd.DataFrame(columns=columns)
        df1['名称'] = indexs
        df2 = pd.DataFrame(columns=columns)
        df2['名称'] = indexs2
        value = metrics.accuracy_score(y, y1), metrics.precision_score(y, y1), metrics.recall_score(y, y1)
        df1.loc[:,'正确百分比'] = value
        conf = metrics.confusion_matrix(y, y1)
        value = conf[1,1]/conf.sum(axis=1)[1], conf[0,0]/conf.sum(axis=1)[0], metrics.f1_score(y, y1)
        df2.loc[:,'正确百分比'] = value
        return df1, df2

    def showinfo(self, y, ypred, bin_num=1):
        pass


class MagicE:
    def __init__(self):
        self.est = self.init_est()

    @abstractmethod
    def init_est(self):
        pass

    @abstractmethod
    def trainpreprocess(self, df_cat, df_num, **kwargs):
        pass
    
    @abstractmethod
    def testpreprocess(self, df_cat, df_num, **kwargs):
        pass

    def set_params(self, **kwargs):
        '''设定模型的参数，用于用户自定义模型参数。'''
        self.est.set_params(**kwargs)

    def select_cat(self, X, columns=None):
        '''选取类别特征，默认所有的object类型的特征为类别，也可以自定义一些特征
        作为类别特征。'''
        result = X.select_dtypes(include='object')
        if columns is not None:
            columns = result.columns|set(columns)
        return columns

    def delete_columns(self, X, columns=None):
        '''删除不满足条件的特征，这些特征将不用于后续的处理和建模。'''
        result = X.copy()
        if columns is not None:
            result = result.drop(columns, axis=1)
        return result
    
    def get_cat_num(self, X, drop_col=None, cat_col=None):
        '''将数据集的特征分为数值特征和类别特征，在不同的模型中采用不同的处理方式，
        默认情况下所有的object类型都为类别特征，所有int和float类型为数值特征，用户
        可以自身知识设置cat_col将一部分数值特征加入到类别特征中'''
        result = self.delete_columns(X, drop_col)
        cat_col = self.select_cat(result, cat_col)
        num_col = result.columns.difference(cat_col)
        return num_col, cat_col

    def get_train(self, X, y=None, drop_col=None, cat_col=None):
        '''得到最终用于建模的数据。'''
        if y is None:
            self.num_col, self.cat_col = self.get_cat_num(X, drop_col, cat_col)
            df_num, df_cat = X.loc[:, self.num_col], X.loc[:,self.cat_col]
            Xtrain = self.trainpreprocess(df_cat, df_num)
        else:
            if hasattr(self, 'num_col') and hasattr(self, 'cat_col'):
                df_cat = X.loc[:, self.cat_col].copy()
                df_num = X.loc[:, self.num_col].copy()
                Xtrain = self.testpreprocess(df_cat, df_num)
            else:
                raise ValueError('模型没有训练，不能用于测试！')
        return Xtrain
            
    def fit(self, X, y, drop_col=None, cat_col=None):
        Xtrain = self.get_train(X, y, drop_col, cat_col)
        self.est = self.est.fit(Xtrain, y)
        return self

    def predict_proba(self, X):
        Xtest = self.get_train(X)
        result = self.est.predict_proba(Xtest)
        return result

    def nocvtrain(self, X, y, drop_col=None, cat_col=None, imbalace=False):
        rainX, validX, trainy, validy = train_test_split(X, y, train_size=0.8)
        Xtrain = self.get_train(trainX, trainy, drop_col, cat_col)
        Xvalid = self.get_train(validX)
        self.est.fit(Xtrain, trainy)

    def train(self, X, y, drop_col=None, cat_col=None, cv=None, imbalance=False):
        if cv is None:
            trainX, validX, trainy, validy = train_test_split(X, y, train_size=0.8)
            Xtrain = self.get_train(trainX, trainy, drop_col, cat_col)
            Xvalid = self.get_train(validX)
            self.est.fit(Xtrain, trainy)
            


class XGBME(MagicE):
    def init_est(self):
        pass

class DartME(MagicE):
    def init_est(self):
        pass

class GossME(MagicE):
    def init_est(self):
        pass

class LogitME(MagicE):
    def init_est(self):
        pass

    def fillnum(self, X, method='mean'):
        X = X.copy()
        if method == 'mean':
            X = X.fillna(X.mean())
        elif method == 'median':
            X = X.fillna(X.median())
        elif method == 'mode':
            X = X.fillna(X.mode().iloc[0])
        return X

    def fillcat(self, X):
        df_cat = X.copy()
        if df_cat.isnull().any().any():
            df_cat = df_cat.fillna('None')
        return df_cat

    def onehot(self, df_cat):
        df_cat = self.fillcat(df_cat)
        enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
        enc.fit(df_cat)
        return enc

    def labelenc(self, df_cat):
        df_cat = self.fillcat(df_cat)
        encs = {}
        for feature in df_cat.columns:
            enc = LabelEncoder()
            enc.fit(df_cat[feature])
            encs[feature] = enc
        return encs

class CartME(MagicE):
    def init_est(self):
        pass
