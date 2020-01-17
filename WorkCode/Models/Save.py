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


from pyecharts import options as opts
from pyecharts.charts import Geo, Map
from pyecharts.globals import GeoType, SymbolType
from pyecharts import render
from pyecharts.globals import ThemeType 



class GeoVisual:
    def __init__(self):
        self.type_ = {'scatter':GeoType.SCATTER, 'heatmap':GeoType.HEATMAP,
        'line':GeoType.LINES, 'effectscatter':GeoType.EFFECT_SCATTER}

    def process(self, series):
        dic = series.to_dict()
        return list(dic.items)

    def show_china(self, series, type_='scatter', title=None):
        data = self.process(series)
        g = (Geo().add_schema(maptype='china', is_roam=False)
                  .add(series_name=series.name, data_pair=data, type_=self.type_[type_])
                  .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                  .set_global_opts(visualmap_opts=opts.VisualMapOpts(),
                                   title_opts=opts.TitleOpts(title=title),))
        return g 

    def show_province(self, name, series, type_='scatter', title=None):
        data=self.process(series)
        g = (Geo().add_schema(maptype=name, is_roam=False)
                  .add(series_name=series.name, data_pair=data, type_=self.type_[type_])
                  .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                  .set_global_opts(visualmap_opts=opts.VisualMapOpts(),
                             title_opts=opts.TitleOpts(title=title),))
        return g

class MapVisual:

    def process(self, series):
        dic = series.to_dict()
        return list(dic.items)

    def map_heatmap(self, series):
        data = self.process(series)
        c = (Map()
            .add(series_name=series.name, data_pair=data, maptype="china")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Map-不显示Label"),
            visualmap_opts=opts.VisualMapOpts(),)
        )
        return c
    def map_scatter(self, series):
        data = self.process(series)
        c = (Map()
            .add(series_name=series.name, data_pair=data, maptype="china")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Map-不显示Label"),)
        )
        return c


from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

class TableVisual:
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

    def gridshow(self, graphs):
        pass


    
