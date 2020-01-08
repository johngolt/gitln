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
from sklearn import clone


class ModelME:
    def __init__(self):
        self.est = self.init_est()
        self.name = self.est.__class__.__name__

    @abstractmethod
    def init_est(self, imbalance=False):
        '''初始化模型，由子类实现具体的模型。'''
        pass

    def set_params(self, **kwargs):
        '''设置模型的参数。'''
        self.est.set_params(**kwargs)

    def get_label(self, data):
        '''计算评价函数的辅助函数，从xgboost和lightgbm的数据结构中得到y值'''
        label = data.get_label()
        return pd.Series(label)

    def _ks(self, y, preds):
        '''计算KS值'''
        fpr, tpr, _ = metrics.roc_curve(y, preds)
        return np.abs(fpr - tpr).max()

    def _lift(self, y, preds):
        '''计算提升度。'''
        groupor = pd.qcut(preds, q=10, duplicates='drop')
        result = y.groupby(groupor).mean() / y.mean()
        return result.max()

    @abstractmethod
    def lift(self, preds, data):
        pass

    @abstractmethod
    def ks(self, preds, data):
        pass

    def get_params(self):
        '''用于得到xgboost和lightgbm的train方法的params参数。'''
        if self.name == 'XGBClassifier':
            params = self.est.get_xgb_params()
            params.pop('n_estimators')
            params['eval_metric'] = 'auc'
        elif self.name == 'LGBMClassifier':
            params = self.est.get_params()
            params.pop('n_estimators')
            params.pop('silent')
            params['metric'] = 'auc'
        return params

    def dataformat(self, X, y):
        '''将训练数据转化为xgboost和lightgbm使用的数据格式，主要用于train方法。'''
        if self.name == 'XGBClassifier':
            data = xgb.DMatrix(X, y)
        elif self.name == 'LGBMClassifier':
            data = lgb.Dataset(X, y)
        return data

    def train_valid_split(self, X, y, imbalance=False):
        '''将训练集进行划分，分为训练集和验证集，在验证集上进行超参数的选择。'''
        if imbalance:
            Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.8, random_state=10)
        else:
            Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.8, random_state=10, stratify=y)
        return Xt, Xv, yt, yv

    def get_train_valid(self, X, y, imbalance=False):
        Xt, Xv, yt, yv = self.train_valid_split(X, y, imbalance)
        tdata = self.dataformat(Xt, yt)
        vdata = self.dataformat(Xv, yv)
        return tdata, vdata

    def fit(self, X, y, imbalance=False):  # 模型训练
        if not hasattr(self, 'train_result'):
            self.train(X, y, imbalance)
        self.estimators = {}
        for key in self.train_result['valid'].keys():
            n = np.argmax(self.train_result['valid'][key])
            est = clone(self.est)
            est = est.set_params(**{'n_estimators':n})
            est = est.fit(X, y)
            self.estimators[key] = est
        return self

    def logit(self, res):
        if res.ndim == 2:
            return res, 1/(1+np.exp(-res.sum(axis=1)))
        else:
            return res, None

    def predict(self, X, best='auc', pred_contrib=False):  # 模型预测
        if best in self.estimators:
            if self.name == 'LGBMClassifier':
                res = self.estimators[best].predict_proba(X,
                                                          pred_contrib=pred_contrib)
                result = self.logit(res)
            elif self.name == 'XGBClassifier':
                data = xgb.DMatrix(X)
                res = self.estimators[best].get_booster().predict(data, pred_contribs=pred_contrib)
                result = self.logit(res)
            return result
        else:
            raise KeyError('{}'.format(best))

    def train(self, X, y, imbalance=False):
        '''用于对模型效果的展示，展示模型在训练集和测试集上的效果。主要针对于提升树模型,返回训练集和验证集上
        评价指标的值和在训练集的模型。'''
        tdata, vdata = self.get_train_valid(X, y, imbalance)
        params = self.get_params()
        res, res1 = {}, {}
        if self.name == 'XGBClassifier':
            est = xgb.train(params,
                            tdata,
                            evals=[(tdata, 'train'), (vdata, 'valid')],
                            num_boost_round=200,
                            feval=self.ks,
                            evals_result=res,
                            verbose_eval=False)
            est = xgb.train(params,
                            tdata,
                            evals=[(tdata, 'train'), (vdata, 'valid')],
                            num_boost_round=200,
                            feval=self.lift,
                            evals_result=res1,
                            verbose_eval=False)
        elif self.name == 'LGBMClassifier':
            est = lgb.train(params,
                            tdata,
                            num_boost_round=200,
                            valid_sets=[tdata, vdata],
                            valid_names=['train', 'valid'],
                            feval=self.ks,
                            verbose_eval=False,
                            evals_result=res)
            est = lgb.train(params,
                            tdata,
                            num_boost_round=200,
                            valid_sets=[tdata, vdata],
                            valid_names=['train', 'valid'],
                            feval=self.lift,
                            verbose_eval=False,
                            evals_result=res1)
        res['train'].update(res1['train'])
        res['valid'].update(res1['valid'])
        self.train_result = res
        self.train_est = est

    def trainreport(self, X, y, imbalance=False):
        self.train(X, y, imbalance)
        valid = pd.DataFrame(self.train_result['valid'])
        limit = valid.idxmax().to_dict()
        Xt, Xv, yt, yv = self.train_valid_split(X, y, imbalance)
        tdata, vdata = self.get_train_valid(X, y, imbalance)
        report = {}
        for key in limit.keys():
            if self.name == 'XGBClassifier':
                typred = 1 - self.train_est.predict(tdata, ntree_limit=limit[key])
                vypred = 1 - self.train_est.predict(vdata, ntree_limit=limit[key])
            elif self.name == 'LGBMClassifier':
                typred = 1 - self.train_est.predict(Xt, ntree_limit=limit[key])
                vypred = 1 - self.train_est.predict(Xv, ntree_limit=limit[key])
            report[key] = [typred, yt, vypred, yv]
        return self.train_result, report


class XGBME(ModelME):  # xgboost
    def init_est(self):
        params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel':1,
               'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0,
               'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 5,
                'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0,
                'reg_lambda': 0, 'scale_pos_weight': 1,'subsample': 0.5 }
        est = xgb.XGBClassifier(**params)
        return est

    def ks(self, preds, data):
        y = self.get_label(data)
        res = self._ks(y, preds)
        return 'KS', res

    def lift(self, preds, data):
        y = self.get_label(data)
        res = self._lift(y, preds)
        return 'Lift', res


class LGBME(ModelME):  # lightgbm
    def init_est(self):
        params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.5,
                    'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': 5,
                    'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
                    'n_estimators': 200, 'n_jobs': -1, 'num_leaves': 10, 'reg_alpha': 0.0,
                    'reg_lambda': 0.0, 'subsample': 0.5, 'subsample_for_bin': 200000,
                    'subsample_freq': 5}
        est = lgb.LGBMClassifier(**params)
        return est

    def ks(self, preds, data):
        y = self.get_label(data)
        res = self._ks(y, preds)
        return ('KS', res, True)

    def lift(self, preds, data):
        y = self.get_label(data)
        res = self._lift(y, preds)
        return ('Lift', res, True)


class ValidResult:

    def get_best_indicator(self, res):
        train = pd.DataFrame(res['train'])
        valid = pd.DataFrame(res['valid'])
        result = {}
        for key in res['valid'].keys():
            idx = valid[key].idxmax()
            temp1 = train.loc[idx, :]
            temp2 = valid.loc[idx, :]
            res = pd.concat([temp1, temp2], axis=1)
            res.columns = ['训练集', '测试集']
            result[key] = res
        return result

    def get_group(self, y, ypred):
        category, bins = pd.qcut(ypred, q=10, duplicates='drop', retbins=True)
        cate = category.set_categories(bins[:-1], rename=True)
        return cate, bins[:-1][::-1]

    def bin_info(self, y, ypred):
        columns = ['箱', '分数', '样本数量', '关注样本数', '区间关注类别比例',
        '累计关注类别比例', '区间准确率', '累计准确率', '提升', '累计提升']
        cate, bins = self.get_group(y, ypred)
        df = pd.DataFrame(columns=columns, index=bins)
        group = y.groupby(cate)
        df['箱'] = range(1, len(df)+1)
        df['分数'] = df.index 
        df['样本数量'] = group.size()
        df['关注样本数'] = group.sum()
        df['区间关注类别比例'] = df['关注样本数']/y.sum()
        df['累计关注类别比例'] = df['区间关注类别比例'].cumsum()
        df['区间准确率'] = df['关注样本数']/df['样本数量']
        df['累计准确率'] = (df['关注样本数'].cumsum())/df['样本数量'].cumsum()
        df['提升'] = df['区间准确率']/y.mean()
        df['累计提升'] = df['累计准确率']/y.mean()
        return df

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
        return df1, df2


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