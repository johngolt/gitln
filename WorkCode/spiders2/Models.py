from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import minepy
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import RFECV, GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier

def split_cat_num(data, cat=15):
        '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其
    归为类别特征中。'''
        categorical = data.select_dtypes(include='object')
        numerical = data.select_dtypes(exclude='object')
        nunique = numerical.nunique().sort_values(ascending=True)
        n_index = nunique[nunique<cat].index
        num = numerical.columns.difference(n_index)
        category = categorical.columns
        return category, num, n_index

class Constant:
    
    def __init__(self, delete=0.9, threshold2=2, bins=0.6):
        '''初始化，threshold1为过滤的阈值，单个值所占的比例；
        threshold2为不同取值下目标分布是否存在显著差异的阈值。'''
        self.delete = delete
        self.threshold2 = threshold2
        self.bins = bins
        self.drop = None
        
    def check_constant(self, data):
        '''检测常变量，如果特征为常变量则删除。'''
        nuniq = data.nunique(dropna=False)
        drop_columns = nuniq[nuniq==1].index
        self.drop = drop_columns
        res = data.drop(drop_columns, axis=1)
        return res
    
    def most_frequent(self, data):
        '''计算每个特征中出现频率最高的项所占的比例和对应的值'''
        data = self.check_constant(data)
        records_count = data.shape[0]
        col_most_values,col_large_value = {},{}
        
        for col in data.columns:
            value_counts = data[col].value_counts()
            col_most_values[col] = value_counts.max()/records_count
            col_large_value[col] = value_counts.idxmax()
        # 使用字典生产DataFrame
        col_most_values_df = pd.DataFrame.from_dict(col_most_values, 
                                                    orient = 'index')
        col_most_values_df.columns = ['max percent']
        col_most_values_df = col_most_values_df.sort_values(
            by = 'max percent', ascending = False)
        return col_most_values_df, col_large_value
    
    def plot_frequency(self, data, N = 30):
        '''绘制出现的频率图'''
        from bokeh.plotting import figure, output_notebook, show
        from bokeh.models import ColumnDataSource
        output_notebook()
        col_most_values_df,_ = self.most_frequent(data)
        source = ColumnDataSource(col_most_values_df[:N])
        p = figure(x_range=col_most_values_df.index[:N].to_list(),
                   tooltips=[('x','$x'),('percent','@{max percent}')])
        p.xaxis.major_label_orientation='vertical'
        p.vbar('index',width=0.5,top='max percent', source=source)
        show(p) 
        
    def frequence_bin(self,data):
        '''进行0-1编码的特征'''
        col_most,col_large = self.most_frequent(data)
        masked = np.logical_and(col_most['max percent']<self.delete,
                               col_most['max percent']>=self.bins)
        cols = list(col_most[masked].index)
        self.bins_cols = cols
        for each in cols:
            data[each+'_bins'] = (data[each]==col_large[each]).astype(int)
        return data
    
    def frequence_diff(self, data, y):
        '''计算多数值占比过高的字段中，少数值的坏样本率是否会显著高于多数值'''
        col_most, col_large = self.most_frequent(data)
        large_percent_cols = list(
             col_most[col_most['max percent']>=self.delete].index)
        bad_rate_diff = {}
        for col in large_percent_cols:
            large_value = col_large[col]
            temp = data.loc[:,[col]]
            temp = pd.concat([temp, y], axis=1)
            temp[col] = (temp[col] == large_value).astype(int)
            bad_rate = temp.groupby(col).mean()
            if bad_rate.iloc[0, 0] == 0:
                bad_rate_diff[col] = 0
                continue
            lograte = bad_rate.iloc[0,0]/bad_rate.iloc[1,0]
            bad_rate_diff[col] = np.log(lograte)
        if bad_rate_diff == {}:
            return bad_rate_diff
        bad_rate_diff = pd.DataFrame.from_dict(bad_rate_diff, 
                                               orient='index')
        bad_rate_diff.columns=['diffence']
        return bad_rate_diff

    def plot_bad_rate_diff(self, data,y):
        '''绘制差异图'''
        bad_rate_diff = self.frequence_diff(data, y)
        if isinstance(bad_rate_diff, dict):
            return 
        bad_rate_diff_sorted = bad_rate_diff.sort_values(by='diffence')
        bad_rate_diff_sorted.plot.bar(figsize=(9,12))
        plt.show()
        
    def delete_frequency(self, data, y):
        '''对于没有明显差异，同时特征中某个值出现的概率很高的特征进行删除。 '''
        data = self.check_constant(data)
        bad_rate_diff = self.frequence_diff(data, y)
        _, col_large = self.most_frequent(data)
        if isinstance(bad_rate_diff,dict):
            return data
        bad_rate_diff = bad_rate_diff.abs()
        drop_columns = bad_rate_diff[bad_rate_diff['diffence']<self.threshold2].index
        for each in bad_rate_diff.index:
            if each in drop_columns:
                continue
            else:
                data[each+'_bins'] = (data[each]==col_large[each]).astype(int)
        res = data.drop(bad_rate_diff.index, axis=1)
        self.drop_columns = drop_columns  # 删除的特征
        self.bad_rate_columns = bad_rate_diff.index  # 所有满足的特征
        self.col_large = col_large
        return res
    
    def __call__(self, data, y=None):
        if y is None:
            data = data.drop(self.drop,axis=1)
            for each in self.bad_rate_columns:
                for each in self.drop_columns:
                    continue
                else:
                    data[each+'_bins'] = (data[each]==self.col_large[each]).astype(int)
            res = data.drop(self.bad_rate_columns, axis=1)
            return res
        else:
            return self.delete_frequency(data, y)


        
class Missing:
    
    def __init__(self, delete=0.9, indicator=0.6, fill=0.1,cat=15):
        '''初始化三个阈值,删除的阈值，产生indicator的阈值，填充的阈值。'''
        self.delete = delete
        self.indicator = indicator
        self.fill = fill
        self.cat = cat
        self.delete_index = None
        self.indicator_index = None
        self.fill_index = None
        self.fill_value={}
        
    def is_null(self, data):
        null = data.isnull()
        return null
    
    def find_index(self, data, method='delete'):
        '''找到满足条件的特征'''
        if method=='delete': 
            threshold = self.delete
        elif method == 'indicator':
            threshold = self.indicator
        else:
            threshold = self.fill
        length = data.shape[0]
        null = self.is_null(data)
        ratio = null.sum()/length
        index = ratio[ratio>threshold].index
        return index
    
    def delete_null(self, data,row=True):
        '''删除缺失比例较高的特征，同时将缺失比例较高的样本作为缺失值删除。'''
        index = self.find_index(data)
        self.delete_index=index
        data = data.drop(index, axis=1)
        if row:
            data = self.delete_items(data)
        return data
    
    def delete_items(self, data,value=None):
        '''删除包含缺失值较多的样本。'''
        null = self.is_null(data).sum(axis=1)
        if value is None:
            per1, per3 = null.quantile([0.25, 0.75])
            if per1 == per3:
                return data
            value = 2*per3 - per1
        index2 = null[null>value].index
        data = data.drop(index2)
        return data
    
    def indicator_null(self, data, row=True):
        '''生产特征是否为缺失的指示特征，同时删除原特征。'''
        data = self.delete_null(data, row=row)
        index = self.find_index(data, method='indicator')
        self.indicator_index = index
        for each in index:
            data['is_'+each] = pd.isnull(data[each]).astype(int)
        data = data.drop(index, axis=1)
        return data
    
    
    def another_class(self, data, row=True):
        '''对于类别特征而言，所有缺失值另作一类；对于数值特征而言，使用中位数
        填补，同时生产一个是否为缺失值的指示特征。'''
        data = self.indicator_null(data, row=row)
        cat, num, n_index = split_cat_num(data, self.cat)
        data.loc[:,cat] = data.loc[:,cat].fillna('None')
        data.loc[:,n_index] = data.loc[:,n_index].fillna(data.loc[:,n_index].max().astype(int)+1)
        index = self.find_index(data.loc[:,num], method='fill')
        self.fill_index = index
        for each in index:
            data['is_'+each] = pd.isnull(data[each]).astype(int)
            self.fill_value[each] = data[each].median()
            data[each] = data[each].fillna(data[each].median())
        return data
    
    def fill_null(self, data, row=True):
        '''使用中位数填补缺失值'''
        data = self.another_class(data, row=row)
        self.fill_value.update(data.median().to_dict())
        data = data.fillna(data.median())
        return data
    
    def __call__(self, data, trainable=True, row=True):
        if trainable:
            return self.fill_null(data, row=row)
        else:
            data = data.drop(self.delete_index, axis=1)
            for each in self.indicator_index:
                data['is_'+each] = pd.isnull(data[each]).astype(int)
            data = data.drop(self.indicator_index, axis=1)
            cat, num, n_index = self.split_cat_num(data)
            data.loc[:,cat] = data.loc[:,cat].fillna('None')
            data.loc[:,n_index] = data.loc[:,n_index].fillna(1e8)
            for each in self.fill_index:
                data['is_'+each] = pd.isnull(data[each]).astype(int)
            data = data.fillna(self.fill_value)
            return data


class Encoding:
    
    def __init__(self,threshold=15):
        self.threshold = threshold
        self.cat = None
        self.n_index = None
        self.num = None
        self.encs = {}
        
    def onehot(self, data,trainable=True,method='cat'):
        if data.empty:
            return data
        names = data.columns
        features={'x{}'.format(i):names[i] for i in range(len(names))}
        if trainable:
            enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
            self.encs[method] = enc
            enc.fit(data)
        else:
            enc = self.encs[method]
        pre_columns = enc.get_feature_names()
        columns = [features[each.split('_',1)[0]]+'_'+each.split('_',1)[1]
                   for each in pre_columns]
        res = enc.transform(data).toarray()
        result = pd.DataFrame(res,columns=columns)
        return result
    
    def __call__(self, data, trainable=True):
        if trainable:
            self.cat, self.num, self.n_index = self.split_cat_num(data)                
            cates = self.onehot(data.loc[:,self.cat],method='cat')
            n_indexs = self.onehot(data.loc[:,self.n_index],
                                   method='n_index')
            data = pd.concat((cates, n_indexs, 
                               data.loc[:,self.num]),axis=1)
        else:
            cates = self.onehot(data.loc[:,self.cat],trainable=trainable,
                                method='cat')
            n_indexs = self.onehot(data.loc[:,self.n_index],
                                   trainable=trainable,method='n_index')
            data = pd.concat((cates, n_indexs, 
                               data.loc[:,self.num]),axis=1)
        return data

def decorate(f):
    def inner(self,features, target=None):
        if isinstance(features,np.ndarray):
            result = f(self,features,target)
        else:
            self.columns = features.columns
            if target is None:
                result = f(self,features.values,target)
            else:
                result = f(self,features.values,target.values)
        return result
    return inner

class FeatureRelation:
    def __init__(self):
        self.columns = None
    @decorate
    def pearson(self, X, y=None):
        m,n = X.shape
        if y is None:
            result = []
            for i in range(n-1):
                for j in range(i+1,n):
                    value,_ = stats.pearsonr(X[:,i],X[:,j])
                    result.append(value)
            return squareform(result)+np.identity(n)
        result = [stats.spearmanr(X[:,j], y)[0] for j in range(n)]
        return np.array(result) 
    @decorate
    def spearman(self, X, y=None):
        if y is None:
            result,_ = stats.spearmanr(X)
            return result
        m,n = X.shape
        result=[stats.spearmanr(X[:,j],y) for j in range(n)]
        return np.array(result)
    @decorate
    def mic(self, X, y=None):
        X=X.transpose(1,0)
        if y is None:
            mic = minepy.pstats(X)+np.identity(X.shape[1])
            return mic
        y = y[None,:]
        result,_ = minepy.cstats(X,y)
        return result.ravel()
    def _distcorr(self, X, Y):
        X = X[:, None]
        Y = Y[:, None]
        n = X.shape[0]
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor
    @decorate
    def distcorr(self, X,y=None):
        m,n = X.shape
        if y is None:
            result = []
            for i in range(n-1):
                for j in range(i+1,n):
                    value = self._distcorr(X[:,i],X[:,j])
                    result.append(value)
            return squareform(result)+np.identity(n)
        result = [self._distcorr(X[:,j], y) for j in range(n)]
        return np.array(result)
    def plot_corr(self,data,target=None):
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.set_ylabel('Correlogram')
        if target is None:
            data = pd.DataFrame(data,columns=self.columns,
                                index=self.columns)
            sns.heatmap(data, xticklabels=data.columns, 
            yticklabels=data.columns, cmap='RdYlGn',
            center=0, annot=True,ax=ax)
        else: 
            data = pd.Series(data, 
                             index=self.columns).sort_values()
            data=data.reset_index()
            data.columns=['a','b']
            ax.vlines(x=data.index, ymin=0, ymax=data['b'], 
                      color='firebrick', alpha=0.7, linewidth=2)
            ax.scatter(x=data.index, y=data['b'], s=75, 
                       color='firebrick', alpha=0.7)
            ax.set_title('Correlogram', 
                         fontdict={'size':22})
            ax.set_xticks(data.index)
            ax.set_xticklabels(data['a'].str.upper(), 
                    rotation=60, 
                fontdict={'horizontalalignment': 'right', 'size':12})
            for row in data.itertuples():
                ax.text(row.Index, row.b*1.01, s=round(row.b, 2),
                        horizontalalignment= 'center',
                        verticalalignment='bottom', fontsize=14)

class FeatureSelection:
    def __init__(self,to_select=None):
        self.estimator = RandomForestClassifier(n_estimators=10,
                                                max_depth=4)
        self.score = 'roc_auc'
        self.masked = None
        
    def select_by_filter(self, func, X, y):
        to_select = X.shape[1]//2
        gus = GenericUnivariateSelect(func, 'k_best', 
                                              param=to_select)
        gus.fit(X,y)
        self.masked = gus.get_support()
    
    def select_by_wrapper(self, X, y):
        step = max(1,X.shape[1]//50)
        to_select = X.shape[1]//2
        clf = RFECV(self.estimator,step=step,
                min_features_to_select=to_select,scoring=self.score,
                   cv=5, n_jobs=-1)
        clf.fit(X,y)
        self.masked = clf.support_
        
    def __call__(self, X, y=None,func=None):
        if y is None:
            return X.loc[:,self.masked]
        else:
            if func is None:
                self.select_by_wrapper(X, y)
                return X.loc[:,self.masked]
            else:
                self.select_by_filter(func,X,y)
                return X.loc[:,self.masked]


class Outlier:
    def __init__(self, method=None, cat=15):
        self.method = method
        self.cat = cat
    def split_cat_num(self, data):
        '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其
    归为类别特征中。'''
        categorical = data.select_dtypes(include='object')
        numerical = data.select_dtypes(exclude='object')
        nunique = numerical.nunique().sort_values(ascending=True)
        n_index = nunique[nunique<self.cat].index
        num = numerical.columns.difference(n_index)
        category = categorical.columns
        return category, num, n_index
    
    def plot_data(self, data, target):
        _,num, _ = self.split_cat_num(data)
        data = pd.concat([data, target], axis=1)
        name = target.name
        melt = pd.melt(data, id_vars=name, 
                       value_vars = [f for f in num])
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, 
                          sharex=False, sharey=False)
        g.map(sns.stripplot, name, 'value', jitter=True,
              palette="muted")
    def __call__(self, data, target):
        self.plot_data(data, target)


class Model:
    
    def __init__(self, estimator, params, cv=5, score=None,
                 imbalance=False):
        self.estimator = estimator #模型
        self.params = params  # 调节参数,字典
        if imbalance:
            self.cv = StratifiedKFold(cv)
        else:
            self.cv=cv
        self.score = score # 评分函数
        self.res = {} 
        self.temp = self.estimator.get_params()
    
    def isequal(self):
        if not(isinstance(self.res,dict) and isinstance(self.temp,dict)):
            return False
        keys = set(self.res.keys()).union(self.temp.keys())
        for key in keys:
            if self.res.get(key,-1) != self.temp.get(key, -1):
                return False
        return True
    
    def search(self, X, y):
        i = 0
        while i<=20 and not(self.isequal()):
            self.res = self.temp
            for each in self.params:
                grid = GridSearchCV(self.estimator,
                            param_grid={each:self.params[each]},
                            scoring=self.score, cv=self.cv,iid=True,
                                   n_jobs=-1)
                grid.fit(X, y)
                self.temp = grid.best_estimator_.get_params()
                self.estimator = grid.best_estimator_
            i += 1
        return self.estimator, self.temp, grid.best_score_
    
    def is_search(self, X, y, search=True):
        if search:
            estimator,*_ = self.search(X, y)
            return estimator
        else:
            return self.estimator
        
    def get_fpr_tpr(self, X, y, trainable=True, search=True):
        if trainable:
            estimator = self.is_search(X, y, search=search)
            estimator.fit(X, y)
            y_pred = estimator.predict_proba(X)[:,1]
            fpr,tpr,threshold = roc_curve(y, y_pred,
                                      drop_intermediate=False)
        else:
            y_pred = self.estimator.predict_proba(X)[:,1]
            fpr,tpr,threshold = roc_curve(y, y_pred,
                                      drop_intermediate=False)
        return fpr, tpr, threshold
    
    def get_auc_ks(self, X, y, trainable=True, search=True):
        fpr, tpr, threshold = self.get_fpr_tpr(X, y, trainable, search)
        roc_auc = auc(fpr,tpr)
        ks = (tpr-fpr).max()
        return roc_auc, ks
     
    def __call__(self, X, y, trainable=True, search=True):
        return self.get_auc_ks(X,y, trainable, search)
    
    def plot_aoc_ks(self, X, y,trainable=True, search=True):
        fpr, tpr, threshold = self.get_fpr_tpr(X, y, trainable, search)
        roc_auc = auc(fpr,tpr) # 计算AUC值
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        ax1.plot(fpr,tpr,color='darkorange', lw=2, 
                       label='ROC Curve(area=%0.2f)'%roc_auc)
        ax1.plot([0,1], [0,1], color='navy',lw=2,linestyle='--')
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1.05)
        ax1.set_xlabel('False Positive Rate')
        ax1.set(xlabel='False Positve Rate',ylabel='True Positive Rate',
                         title='ROC')
        ax1.legend(loc='lower right')
        index = (tpr-fpr).argmax()
        ax2.plot(threshold, 1-tpr, label='tpr')
        ax2.plot(threshold, 1-fpr, label='fpr')
        ax2.plot(threshold, tpr-fpr, label='KS %0.2f'%(tpr-fpr)[index],
                     linestyle='--')
        ax2.vlines(threshold[index],1-fpr[index],1-tpr[index])
    
        ax2.set(xlabel='score')
        ax2.set_xlim(0,1)
        ax2.set_title('KS Curve')
        ax2.legend(loc='upper left', shadow=True, fontsize='x-large')
        plt.show( )
