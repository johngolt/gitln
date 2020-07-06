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


def split_cat_num(data, cat=15):
    '''对特征进行分类，得到数值特征和类别特征，对于数值特征中取值较少的特征，将其归为类别特征中。'''
    categorical = data.select_dtypes(include='object')
    numerical = data.select_dtypes(exclude='object')
    nunique = numerical.nunique().sort_values(ascending=True)
    n_index = nunique[nunique < cat].index
    num = numerical.columns.difference(n_index)
    category = categorical.columns
    return category, num, n_index

    
class PlotFunc:
    '''针对类别特征和数值特征可视化的类别，更加具体的可视化特征的一些信息。'''
    def __init__(self):
        self.font1 = {'family': 'Calibri', 'weight': 'normal', 'size': 14}
        self.font2 = {'family': 'Calibri', 'weight': 'normal', 'size': 18}

    def get_ax(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot()
            return ax
        return ax

    def plot_bin(self, data, ax=None):
        ax = self.get_ax(ax)
        data.columns = ['a', 'b']
        ax.vlines(x=data.index, ymin=0,
                  ymax=data['b'], color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=data.index, y=data['b'],
                   s=75, color='firebrick', alpha=0.7)

        ax.set_xticks(data.index)
        ax.set_xticklabels(data['a'],
                           rotation=90,
                           fontdict={
                               'horizontalalignment': 'right',
                               'size': 12
                           })

        for row in data.itertuples():
            ax.text(row.Index,
                    row.b * 1.01,
                    s=round(row.b, 2),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=14)
        return ax