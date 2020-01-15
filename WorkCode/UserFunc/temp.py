import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib import gridspec


class Outlier:
    def ploting_cat_fet(self, df, cols, y):
        '''绘制类别特征，柱状图为每个值在特征中出现的数目及所占的比例，折线图为每个取值
        情况下，其中的坏样本率。'''
        total = len(df)
        # 图形的参数设置
        nrows, ncols = len(cols)//2+1, 2
        grid = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(16, 20*nrows//3))
        df = pd.concat([df, y], axis=1)
        name = y.name

        for n, col in enumerate(cols):
            tmp = pd.crosstab(df[col], df[name],
                              normalize='index') * 100
            tmp = tmp.reset_index()
            tmp.rename(columns={0: 'No', 1: 'Yes'}, inplace=True)

            ax = fig.add_subplot(grid[n])
            sns.countplot(x=col, data=df, order=list(tmp[col].values),
                          color='green')  # 绘制柱状图

            ax.set_ylabel('Count', fontsize=12)  # 设置柱状图的参数
            ax.set_title(f'{col} Distribution by Target', fontsize=14)
            ax.set_xlabel(f'{col} values', fontsize=12)

            gt = ax.twinx()  # 绘制折线图
            gt = sns.pointplot(x=col, y='Yes', data=tmp,
                               order=list(tmp[col].values),
                               color='black', legend=False)

            mn, mx = gt.get_xbound()  # 绘制水平线
            gt.hlines(y=y.sum()/total, xmin=mn, xmax=mx,
                      color='r', linestyles='--')

            gt.set_ylim(0, tmp['Yes'].max()*1.1)  # 设置y轴和title
            gt.set_ylabel("Target %True(1)", fontsize=16)
            sizes = []

            for p in ax.patches:  # 标识每个值所占的比例。
                height = p.get_height()
                sizes.append(height)
                ax.text(p.get_x()+p.get_width()/2.,
                        height + 3,
                        '{:1.2f}%'.format(height/total*100),
                        ha="center", fontsize=12)
            ax.set_ylim(0, max(sizes) * 1.15)
        plt.subplots_adjust(hspace=0.5, wspace=.3)
