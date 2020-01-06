import numpy as np
import pandas as pd
from abc import abstractmethod
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
import xgboost as xgb
import lightgbm as lgb


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
    
    def line(self, train, valid, title):
        index = int(np.argmax(valid)) 
        x = list(range(len(train)))
        c = (Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(x).add_yaxis("Train",ldic['train']['auc'],symbol='none',)
            .add_yaxis("Valid", ldic['valid']['auc'], symbol='none',
                    markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='auc', x=index)],
                    label_opts=opts.LabelOpts(is_show=False)))
            .set_global_opts(title_opts=opts.TitleOpts(title=title),
                            xaxis_opts=opts.AxisOpts(type_="value")))
        return c

    def tabshow(self, res):
        trains, valids = res['train'], res['valid']
        tab = Tab(page_title='模型选择')
        for key in trains.keys():
            tab.add(self.line(trains[key], valids[key], key), key)
        return tab

    def get_group(self, y, ypred):
        category, bins = pd.qcut(ypred, q=10, duplicates='drop', retbins=True)
        cate = category.set_categories(bins[:-1], rename=True)
        return cate, bins[:-1][::-1]

    def bin_info(self, y, ypred):
        columns = ['箱', '分数', '样本数量', '关注样本数', '区间关注类别比例',
        '累计关注类别比例', '区间准确率', '累计准确率', '尾部准确率',
        '提升', '累计提升']
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
        columns = ['实际类别', '类别总数', '正确百分比', '预测类别 非关注', '预测类别 关注']
        indexs = ['非关注', '关注', '合计']
        df = pd.DataFrame(columns=columns)
        df['实际类别'] = indexs
        df = df.set_index('实际类别')
        df['类别总数'] = [len(y)-y.sum(), y.sum(), len(y)]

        _, bins = self.get_group(y, ypred)
        y1 = (ypred > bins[bin_num-1]).astype(int)

        df.iloc[:2, -2:] = metrics.confusion_matrix(y, y1)
        df.loc['关注', '正确百分比'] = df.at['关注', '预测类别 关注']/df.at['关注', '类别总数']
        df.loc['非关注', '正确百分比'] = df.at['非关注', '预测类别 非关注']/df.at['非关注', '类别总数']
        df.loc['合计', '预测类别 关注'] = df.at['关注', '预测类别 关注']/df.loc[:, '预测类别 关注'].sum()
        df.loc['合计', '预测类别 非关注'] = df.at['非关注', '预测类别 非关注']/df.loc[:, '预测类别 非关注'].sum()
        return df

    def indicate(self, y, ypred, bin_num=1):
        _, bins = self.get_group(y, ypred)
        y1 = (ypred > bins[bin_num-1]).astype(int)
        columns = ['名称', '正确百分比']
        indexs = ['准确率', '精确率', '召回率']
        indexs2 = ['灵敏度', '特异度', 'F1统计']
        df1 = pd.DataFrame(columns=columns)
        df1['名称'] = indexs
        df2 = pd.DataFrame(columns=columns)
        df2['名称'] = indexs2
        value = metrics.accuracy_score(y, y1), metrics.precision_score(y, y1), metrics.recall_score(y, y1)
        df1.loc[:, '正确百分比'] = value
        conf = metrics.confusion_matrix(y, y1)
        value = conf[1, 1]/conf.sum(axis=1)[1], conf[0, 0]/conf.sum(axis=1)[0], metrics.f1_score(y, y1)
        df2.loc[:, '正确百分比'] = value
        return df1, df


class PreprocessME:
    '''特征处理'''
    @abstractmethod
    def trainpreprocess(self, df_cat, df_num, **kwargs):
        pass
    
    @abstractmethod
    def testpreprocess(self, df_cat, df_num, **kwargs):
        pass

    def select_cat(self, X, columns=None):
        '''选取类别特征，默认所有的object类型的特征为类别，也可以自定义一些特征
        作为类别特征。'''
        result = X.select_dtypes(include='object')
        cat = result.columns
        if columns is not None:
            cat = cat | set(columns)
        return cat

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
            df_num, df_cat = X.loc[:, self.num_col], X.loc[:, self.cat_col]
            Xtrain = self.trainpreprocess(df_cat, df_num)
        else:
            if hasattr(self, 'num_col') and hasattr(self, 'cat_col'):
                df_cat = X.loc[:, self.cat_col].copy()
                df_num = X.loc[:, self.num_col].copy()
                Xtrain = self.testpreprocess(df_cat, df_num)
            else:
                raise ValueError('模型没有训练，不能用于测试！')
        return Xtrain


class LGBPreprocess(PreprocessME):
    def fillcat(self, X):
        df_cat = X.copy()
        if df_cat.isnull().any().any():
            df_cat = df_cat.fillna('None')
        return df_cat

    def labelenc(self, df_cat):
        df_cat = self.fillcat(df_cat)
        result = df_cat.copy()
        encs = {}
        for feature in df_cat.columns:
            enc = LabelEncoder()
            enc.fit(df_cat[feature])
            encs[feature] = enc
            result[feature] = enc.transform(df_cat[feature])
        return encs, result

    def trainpreprocess(self, df_cat, df_num):
        self.encs, result = self.labelenc(df_cat)
        result = pd.concat([result, df_num], axis=1)
        return result

    def testpreprocess(self, df_cat, df_num):
        df_cat = self.fillcat(df_cat)
        cat = df_cat.copy()
        assert hasattr(self, 'encs')
        for feature in self.encs.keys():
            cat[feature] = self.encs[feature].transform(df_cat[feature])
        result = pd.concat([cat, df_num], axis=1)
        return result


class ModelME:
    def __init__(self):
        self.est = self.init_est()
        self.name = self.est.__class__.__name__

    @abstractmethod
    def init_est(self, imbalance=False):
        pass

    def set_params(self, **kwargs):
        '''设置模型的参数。'''
        self.est.set_params(**kwargs)
    
    def get_label(self, data):
        label = data.get_label()
        return pd.Series(label)
        
    def _ks(self, y, preds):
        fpr, tpr, _ = roc_curve(y, preds)
        return np.abs(fpr-tpr).max()
    
    def _lift(self, y, preds):
        groupor = pd.qcut(preds, q=10, duplicates='drop')
        result = y.groupby(groupor).mean()/label.mean()
        return result.max()

    @abstractmethod
    def lift(self, data, preds):
        pass

    @abstractmethod
    def ks(self, data, preds):
        pass

    def get_params(self):
        if self.name == 'XGBClassifier':
            params = self.est.get_xgb_params()
            params.pop('n_estimators')
            params['eval_metric'] = 'auc'
        elif self.name == 'LGBMClassifier':
            params = self.est.get_params()
            params.pop('n_estimators')
            params['metric'] = 'auc'
        return params

    def dataformat(self, X, y):
        if self.name == 'XGBClassifier':
            data = xgb.DMatrix(X, y)
        elif self.name == 'LGBMClassifier':
            data = lgb.Dataset(X, y)
        return data

    def get_train_valid(self, X, y, imbalance=False):
        if imbalance:
            Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.8, random_state=10)
        else:
            Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.8, random_state=10, stratify=y)
        tdata = self.dataformat(Xt, yt)
        vdata = self.dataformat(Xv, yv)
        return tdata, vdata
    
    def fit(self, X, y, imbalance=False):
        est = clone(self, est)
        res, _ self.train(X, y, imbalance)
        self.estimators = {}
        for key in res['valid'].keys():
            n = np.argmax(res['valid'][key])
            est = est.set_params(**{'n_estimators':n})
            est = est.fit(X, y)
            self.estimators[key] = est
        return self
    
    def predict(self, X, best='auc'):
        if best in self.estimators:
            preds = self.estimators[key].predict_proba(X)
        else:
            raise KeyError('{}'.format(best))
    
    def train(self, X, y, imbalance=False):
        tdata, vdata = self.get_train_valid(X, y, imbalance)
        params = self.get_params()
        res, res1 = {}, {}
        if self.name == 'XGBClassifier':
            result = xgb.train(params, tdata, evals=[(tdata, 'train'), (vdata, 'valid')],
                                num_boost_round=200, feval=self.ks, evals_result=res, 
                                verbose_eval=False)
            result = xgb.train(params, tdata, evals=[(tdata, 'train'), (vdata, 'valid')],
                               num_boost_round=200, feval=self.lift, evals_result=res1, 
                               verbose_eval=False)
        elif self.name == 'LGBMClassifier':
            result = lgb.train(params, ltdata, num_boost_round=200,
                               valid_sets=[ltdata, lvdata], valid_names=['train', 'valid'],
                               feval=self.ks, verbose_eval=False, evals_result=res)
            result = lgb.train(params, ltdata, num_boost_round=200,
                               valid_sets=[ltdata, lvdata], valid_names=['train', 'valid'],
                               feval=self.lift, verbose_eval=False, evals_result=res1)
        res['train'].update(res1['train'])
        res['valid'].update(res1['valid'])
        return res, result  


class XGBME(ModelME):
    def init_est(self):
        params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel':1,
               'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0,
               'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 5,
                'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0,
                'reg_lambda': 0, 'scale_pos_weight': 1,'subsample': 0.5 }
        est = xgb.XGBClassifier()
        self.est = est.set_params(**params)

    def ks(self, data, preds):
        y = self.get_label(data)
        res = self._ks(y, preds)
        return 'KS', res
    
    def lift(self, data, preds):
        y = self.get_label(data)
        res = self._lift(y, preds)
        return 'lift', res


class LGBME(ModelME):
    def init_est(self):
        params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.5,
                    'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': 5,
                    'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
                    'n_estimators': 200, 'n_jobs': -1, 'num_leaves': 10, 'reg_alpha': 0.0,
                    'reg_lambda': 0.0, 'subsample': 0.5, 'subsample_for_bin': 200000,
                    'subsample_freq': 5}
        est = lgb.LGBMClassifier()
        self.est = est.set_params(**params)

    def ks(self, data, preds):
        y = self.get_label(data)
        res = self._ks(y, preds)
        return ('KS', res, True)

    def lift(self, data, preds):
        y = self.get_label(data)
        res = self._lift(y, preds)
        return ('Lift', res, True)
    

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

    def onehot(self, df_cat):
        df_cat = self.fillcat(df_cat)
        enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
        enc.fit(df_cat)
        return enc


class CartME(MagicE):
    def init_est(self):
        pass


class ME:
    @abstractmethod
    def init_preprocess(**kwargs):
        pass
    @abstractmethod
    def init_model(**kwargs):
        pass
    def train(**kwargs):
        pass
    def predict(**kwargs):
        pass