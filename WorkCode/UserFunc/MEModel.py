import numpy as np
import pandas as pd
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
import pyecharts.options as opts
from pyecharts.charts import Line, Tab
from pyecharts.globals import ThemeType
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


class ValidResult:
    def show_tree_num(self, res):
        trains, valids = res['train'], res['valid']
        indexes = pd.DataFrame(valids).idxmax().to_dict()
        values = [opts.MarkLineItem(name=tkey,x=indexes[tkey])
                  for tkey in indexes.keys()]
        mark = opts.MarkLineOpts(data=values,
                label_opts=opts.LabelOpts(is_show=False))
        train, valid = trains['auc'], valids['auc']
        x = list(range(len(train)))
        c = (Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(x)
            .add_yaxis("Train",train, symbol='none',
                            label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("Valid", valid,
                       label_opts=opts.LabelOpts(is_show=False),
                       symbol='none')
            .set_series_opts(markline_opts=mark)
            .set_global_opts(xaxis_opts=opts.AxisOpts(type_="value"),
                            tooltip_opts=opts.TooltipOpts(trigger='axis')))
        return c
    
    def process(self, df,index=False):
        df = df.round(3)
        if index:
            df.index.name='名称'
            df = df.reset_index()
        headers = df.columns
        rows = df.to_numpy().tolist()
        return list(headers), rows

    def table(self, df, title=None):
        headers, rows = self.process(df,index=True)
        tb = Table().add(headers, rows).set_global_opts(
                            title_opts=ComponentTitleOpts(title=title))
        return tb

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
    
    def show_best_indicator(self, res):
        result = self.get_best_indicator(res)
        tab = Tab()
        for key in result.keys():
            temp = self.table(result[key], key)
            tab.add(temp, key)
        return tab
        