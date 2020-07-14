'''
这一模块主要包括一些数据处理的基本方法，用于建模过程中的数据分析和处理。主要包括以下几部分。
split_cat_num:将特征进行分类，分为数值特征，类别特征和数值特征中取值较少可能为类别特征的特征。 

FeatureStatistics:主要对特征进行一些简单的信息统计和可视化，按照类别和数值特征两种来进行处理。

PlotFunc:包含了一些绘图的基本处理，方便后续数据可视化。

Categorical:主要针对类别特征的可视化，用于数据预处理阶段。

Numerical:主要针对于数值特征的可视化，用于数据预处理阶段。

Constant:针对于常值特征的处理和可视化。

CalEnt:计算类别特征的条件信息熵，基尼系数和WOE,IV值

Missing:针对于数据处理过程中的缺失值处理，包含了常用的一些缺失值处理方法。

FeatureStability:用于检验训练集和测试集中特征的稳定性，可以作为特征选择的一部分，也可以用于查看
特征在不同类别之间的分布情况。
'''

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 可以显示负号

    
class PlotFunc:

    def get_ax(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot()
            return ax
        return ax
    
    def gridspecplot(self, data, features, func, target=None, **func_kwargs):
        nrows, ncols = len(features)//2+1, 2
        grid = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(16, 20*nrows//3))
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            func(data, each, target=target, ax=ax, **func_kwargs)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
    
    def _binplot(self, data, ax=None):
        ax = self.get_ax(ax)
        data.columns = ['a', 'b']
        ax.vlines(x=data.index, ymin=0,
                  ymax=data['b'], color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'],
                   s=50, color='firebrick', alpha=1)

        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'], rotation=90,
                           fontdict={'horizontalalignment': 'right',
                               'size': 12})
        ax.grid(axis='y', ls='--')
        return ax

    def plot_bin(self, data, ax=None):
        ax = self.get_ax(ax)
        if isinstance(data, pd.core.series.Series):
            data = data.reset_index()
        if data.shape[0] > 30:
            data.columns = ['a', 'b']
            ax.scatter(x=data['a'], y=data['b'], color='blue')
            ax.xaxis.set_ticklabels([])
            ax.grid(axis='y', ls='--')
        else:
            ax = self._binplot(data, ax=ax)
        return ax

    
            