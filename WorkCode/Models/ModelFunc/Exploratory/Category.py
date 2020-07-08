import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec

class CateVisual:

    def get_ax(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot()
            return ax
        return ax

    def dist_visual(self, data, feature, ax=None, **kwarg):
        ax = self.get_ax(ax)
        temp = data[feature].value_counts(dropna=False)
        if temp.shape[0] > 30:
            ax.scatter(x=temp.index, y=temp.values, **kwarg)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.xaxis.set_ticklabels([])
        else:
            temp.plot(kind='bar', **kwarg)
    
    
    
class Categorical(PlotFunc):
    '''对类别特征进行分析，适合取值不多的类别特征。对于取值很多的类别特征可视化效果不好'''

    def _crosstab(self, data, feature, target, ax=None):
        '''查看目标在类别特征的每个值的分布情况。'''
        ax = self.get_ax(ax)
        ct = pd.crosstab(data[feature], data[target], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('{}'.format(feature))
        ax.axhline(1 - data[target].mean(),
                   color="crimson",
                   alpha=0.9,
                   linestyle="--")
        ax.set_ylim(0, 1)
        plt.ylabel('{} Rate'.format(target))
        ax.set_xlabel('')
        return ax

    def plot_crosstab(self, data, features, target):
        nrows = len(features)//4+1
        fig = plt.figure(figsize=(20, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=4)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            self._crosstab(data, each, target, ax=ax)
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

    def plot_dists(self, data, features, target):
        nrows = len(features)
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_dist(data, each, target, [ax, ax1])
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_dist(self, data, feature, target, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.subplots(1, 2)
        _ = self.plot_bar(data, feature, ax[0])
        _ = self.plot_bar_line(data, feature, target, ax[1])

    def plot_bar(self, data, feature, ax1=None):
        '''用条形图展示值分布。在类别特征中，若某个类别所含样品频数特别少，则可以认为是异常值，舍弃。
        若有多个类别的样本频数都较少，可以考虑合并类别。'''
        ax1 = self.get_ax(ax1)
        df = data[feature].value_counts().sort_values()
        df.plot(kind='bar', ax=ax1)
        ax1.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax1.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax1.set_ylim(bottom=0, top=len(data))
        ax1.legend()
        xs = ax1.xaxis.get_ticklocs()
        for x, y in zip(xs, df.values):
            ax1.annotate(s=str(y), xy=(x, y), xytext=(x*0.95, y*1.01))
        return ax1

    def plot_bar_line(self, data, feature, target, ax=None):
        '''可视化类别特征中样本的频率，和目标变量的分布情况。'''
        ax = self.get_ax(ax)
        df = pd.crosstab(data[feature], data[target])
        ax = df.plot(kind='bar', stacked=True, ax=ax, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax.set_ylim(bottom=0, top=len(data))
        ax.legend(loc=2)

        ax.xaxis.set_ticklabels(df.index)
        xs = ax.xaxis.get_ticklocs()
        odds = df[1]/df[0]
        axt = ax.twinx()
        axt.plot(xs, odds, marker="o")
        axt.set_ylabel("odds", fontdict=self.font1, labelpad=6)
        odd = df.sum()[1]/df.sum()[0]
        axt.axhline(odd, color="crimson", alpha=0.9, linestyle="--")
        return ax
