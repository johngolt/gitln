import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


class BaseSetting:
    def __init__(self):
        self.large=22
        self.med = 16
        self.small = 12
        self.params = {'legend.fontsize': self.med,'figure.figsize': (16, 10),
                'axes.labelsize': self.med,
          'axes.titlesize': self.med, 'xtick.labelsize': self.med,
          'ytick.labelsize': self.med,'figure.titlesize': self.large}
        plt.rcParams.update(self.params)

    def set_label(self, xlabel, ylabel, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set(xlabel=xlabel, ylabel=ylabel)


class Visual:
    def getfigax(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        return fig, ax

    def getmap(self, data, x):
        unique = data[x].unique()
        mapping = {key:value for value, key in enumerate(unique)}
        ser = data[x].replace(mapping)
        return ser, mapping
    
    def catelabels(self, mapping, ax, xaxis=True):
        if xaxis:
            ax.xaxis.set_ticks(list(mapping.values()))
            ax.xaxis.set_ticklabels(list(mapping.keys()))
            plt.setp(ax.xaxis.get_ticklabels(), rotation=90)
        else:
            ax.yaxis.set_ticks(list(mapping.values()))
            ax.yaxis.set_ticklabels(list(mapping.keys()))
        return ax

    def scatterwithline(self, data, x, y, hue=None):
        '''了解两个变量如何相互改变'''
        grid = sns.lmplot(x=x, y=y, hue=hue, data=data, height=7, aspect=1.6, robust=True, 
        palette='tab10', scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
        xmin,xmax = data[x].min(),data[x].max()
        ymin, ymax = data[y].min(),data[y].max()
        grid.set(xlim=(xmin*1.01, xmax*1.01), ylim=(ymin*1.01, ymax*1.01))
        return grid

    def jitter(self, data, x, y):
        '''通常，多个数据点具有完全相同的X和Y值。导致相互绘制并隐藏。
        使用抖动点可以直观地看到它们。'''
        _, ax = self.getfigax()
        sns.stripplot(data[x], data[y], jitter=0.25, siz=6, ax=ax, linewidth=.5)
        ax.set_title('Jittered Plots',fontsize=22)
    
    def countjitter(self, data, x, y):
        '''避免点重叠问题的另一个选择是增加点的大小，点的大小越大，周围的点的集中度就越大。'''
        data = data.copy()
        ser, mapping = self.getmap(data, x)
        data[x] = ser
        ser1, mapping1 = self.getmap(data, y)
        data[y] = ser1
        data = data.groupby([x,y]).size().reset_index(name='counts')
        _, ax = self.getfigax()   
        sns.stripplot(data[x], data[y], size=data['counts'], ax=ax)
        ax = self.catelabels(mapping, ax)
        ax = self.catelabels(mapping1, ax, xaxis=False)
        ax.set_title('Counts Jitter Plot', fontsize=22)

    

    
