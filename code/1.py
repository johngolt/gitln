import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin,clone
import scipy.stats as stats 
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve


class AveragingModels(BaseEstimator, ClassifierMixin):
    def __init__(self,models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        predictions = np.column_stack([model.predict(X) 
                                       for model in self.models_])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                axis=1, arr=predictions)  
    
    def predict_proba(self,X):
        predictions = np.array([model.predict_proba(X)
                                      for model in self.models_])
        return np.mean(predictions, axis=0)



'''Test the null hypothesis that two samples have the same underlying
probability distribution.'''

# stats.epps_singleton_2samp()#分类

class TrainTest:
    '''可视化特征在训练集和测试集上的分布，可以用来发现不稳定的特征。也可以用来可视化
    特征在不同类别的特征，用来选取重要特征或者删除不重要特征。通过函数发现训练和测试集分布不一致
    的特征，返回不一致的特征'''
    def __init__(self, threshold_p=0.01, threshold_statistic=0.3):
        self.p_value = threshold_p
        self.statistic = threshold_statistic

    def statistic_test(self, train, test, func=stats.ks_2samp):
        '''Compute the Kolmogorov-Smirnov statistic on 2 samples.
This is a two-sided test for the null hypothesis that 2 independent
samples are drawn from the same continuous distribution. '''
        diff_cols=[]
        for col in train.columns:
            statistics, pvalue = func(train[col].values,test[col].values)
            if pvalue <= self.p_value and np.abs(statistics) > self.statistic:
                diff_cols.append(col)
        return diff_cols

    def get_statistic(self, train, features): #利用连续特征构造新的特征，
        train = train.copy()
        train['mean'] = train[features].mean(axis=1)
        train['std'] = train[features].std(axis=1)
        train['median'] = train[features].median(axis=1)
        train['skew'] = train[features].skew(axis=1)
        train['kurt'] = train[features].kurt(axis=1)
        return train

    def plot_train_test_num(self, train, test, features, target=False):
        label1 = 'target=0' if target else 'train'
        label2 = 'target=1' if target else 'test'
        if len(features) == 1:
            fig = plt.figure(figsize=(16,6))
            plt.title("Distribution of values")
            sns.distplot(train[features],color="green",
                kde=True,bins=120, label=label1)
            sns.distplot(test[features],color="blue",
                kde=True,bins=120, label=label2)
            plt.legend()
        else:
            train, test = train[features].copy(),test[features].copy()
            train, test = self.get_statistic(train,features), self.get_statistic(test, features)
            columns = train.columns
            nrows = len(columns)//4+1
            fig = plt.figure(figsize=(16,4*nrows))
            fig.suptitle('Distribution of values in train and test',fontsize=16)
            grid = gridspec.GridSpec(nrows,4)
            for i,each in enumerate(columns):
                ax = fig.add_subplot(grid[i])
                sns.distplot(train[each],color="green",
                kde=True,bins=120, label=label1,ax=ax)
                sns.distplot(test[each],color="blue",
                kde=True,bins=120, label=label2,ax=ax)
                ax.set_xlabel('')
                plt.legend()
                ax.set_title('{}'.format(each))
    
    def plot_train_test_cat(self, train, test, features, target=False):
        label1 = 'target=0' if target else 'train'
        label2 = 'target=1' if target else 'test'
        train, test = train[features].copy(),test[features].copy()
        train['train']=1
        test['train']=0
        data = pd.concat([train,test])
        columns = data.columns
        nrows = len(columns)//4+1
        fig = plt.figure(figsize=(16,4*nrows))
        fig.suptitle('Distribution of values in train and test',fontsize=16)
        grid = gridspec.GridSpec(nrows,4)
        for i,each in enumerate(columns[:-1]):
            ax = fig.add_subplot(grid[i])
            sns.countplot(each,hue='train',data=data,ax=ax)
            ax.set_xlabel('')
            ax.legend([label1, label2])
            ax.set_title('{}'.format(each))

    def features_by_target(self, X, y):
        train, test = X.loc[y==0, :], X.loc[y==1, :]
        trainN, trainC = train.select_dtypes(exclude='object'), train.select_dtypes(include='object')
        testN, testC = test.loc[:,trainN.columns], test.loc[:, trainC.columns]
        self.plot_train_test_num(trainN, testN, trainN.columns, target=True)
        self.plot_train_test_cat(trainC, testC, trainC.columns, target=True)


def plot_pr_curve(y_true, y_pred):
    AP = metrics.average_precision_score(y_true, y_pred)
    precision, recall,_ = metrics.precision_recall_curve(y_true,y_pred)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    ax.step(recall,precision,color='c', where='post',alpha=0.5)
    ax.fill_between(recall,precision,color='b',alpha=0.2)
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1.),ylim=(0,1.))
    ax.set_title('Average Precision Score:{:.2f}'.format(AP),fontsize=16)
