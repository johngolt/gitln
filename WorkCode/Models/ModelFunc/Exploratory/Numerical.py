class Numerical(PlotFunc):

    def drop_null_item(self, data, feature=None):
        '''丢弃特征中确实的样本。'''
        if feature is None:
            return data[data.notnull()]
        temp = data.loc[data[feature].notnull(), feature]
        return temp

    def _kstest(self, data, feature):
        '''数值特征的正态性检验，确定分布是否符合正态分布。'''
        mean, std = data[feature].mean(), data[feature].std()
        temp = self.drop_null_item(data, feature)
        _, pvalue = stats.kstest(temp, stats.norm(mean, std).cdf)
        if pvalue < 0.05:
            return False
        else:
            return True

    def kstests(self, data, features):
        mask = [self._kstest(data, each) for each in features]
        return mask

    def plot_strips(self, data, features, target, ax=None):
        '''按照label,对数据集中的数值特征绘制stripplot，可以根据图形从中寻找到
        数值特征中的异常值。'''
        nrows, ncols = len(features)//2+1, 2
        grid = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(16, 20*nrows//3))
        for i, feature in enumerate(features):
            ax = fig.add_subplot(grid[i])
            sns.stripplot(target,
                          feature,
                          jitter=True,
                          palette='muted',
                          order=[0, 1],
                          data=data,
                          ax=ax)

    def plot_kde(self, data, feature, ax=None, **kwargs):
        '''绘制数值特征的kernel density estimation，同时采用
        正太分布进行对照。'''
        ax = self.get_ax(ax)
        sample = self.drop_null_item(data, feature).to_numpy()
        ax = sns.distplot(sample, hist=False, fit=stats.norm, ax=ax, **kwargs)
        ax.legend(ax.lines, ['kde', 'norm'], loc=1)
        ax.set_title('{} kde distribution'.format(
            feature), fontdict=self.font1)
        ax.set_ylabel('Probability', fontdict=self.font2, labelpad=6)
        return ax

    def plot_kdes(self, data, features, **kwargs):
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            self.plot_kde(data, each, ax=ax, **kwargs)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_boxs(self, data, features, **kwargs):
        nrows = len(features)//2+1
        fig = plt.figure(figsize=(10, 5*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i])
            _ = self.plot_box(data, each, ax=ax, **kwargs)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_box(self, data, feature, ax=None, **kwargs):
        '''绘制数值特征的箱型图。'''
        ax = self.get_ax(ax)
        sns.boxplot(data[feature], ax=ax, orient='v', **kwargs)
        ax.set_title('{} boxplot'.format(feature), fontdict=self.font2)
        ax.set_ylabel('{}'.format(feature), fontdict=self.font1, labelpad=6)
        return ax

    def plot_hists(self, data, features, target, bins=50):
        nrows = len(features)
        fig = plt.figure(figsize=(8, 4*nrows))
        grid = gridspec.GridSpec(nrows=nrows, ncols=2)
        for i, each in enumerate(features):
            ax = fig.add_subplot(grid[i, 0])
            self.plot_hist(data, each, ax=ax, bins=bins)
            ax1 = fig.add_subplot(grid[i, 1])
            self.plot_hist_line(data, each, target, ax=ax1, bins=bins)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plot_hist(self, data, feature, ax=None, bins=50):
        '''绘制数值特征的条形图'''
        ax = self.get_ax(ax)
        temp = self.drop_null_item(data, feature)
        bins = min(int(np.sqrt(temp.shape[0])), bins)
        _ = ax.hist(data[feature], bins=bins, alpha=0.7)
        ax.set_title('{} distribution'.format(feature), fontdict=self.font2)
        ax.set_xlabel('{}'.format(feature), fontdict=self.font1, labelpad=2)
        ax.set_ylabel('frequency', fontdict=self.font1, labelpad=6)
        return ax

    def plot_hist_line(self, data, feature, target, ax=None, bins=50):
        '''绘制在目标变量下数值特征的条形图，同时查看目标在特征不同区间下的分布。'''
        ax = self.get_ax(ax)
        X0 = data.loc[data[target] == 0, feature]
        X1 = data.loc[data[target] == 1, feature]
        bins = min(int(np.sqrt(min(len(X0), len(X1)))), bins)
        X0 = self.drop_null_item(X0)
        X1 = self.drop_null_item(X1)
        n1, bins1, _ = ax.hist(X0, bins=bins, alpha=0.6, label='Target=0')
        n2, *_ = ax.hist(X1, bins=bins1, alpha=0.6,
                         bottom=n1, label='Target=1')
        ax.set_title('{} distribution in {}'.format(
            target, feature), fontdict=self.font2)
        ax.set_ylabel('Frequency', fontdict=self.font1, labelpad=6)
        ax.legend(loc=2)
        odds = n2.sum()/n1.sum()
        xs, ys = (bins1[:-1]+bins1[1:])/2, (n2+odds)/(n1+1)
        axt = ax.twinx()
        axt.plot(xs, ys, marker='*')
        axt.axhline(odds, color='crimson', alpha=0.8, linestyle='--')
        axt.set_ylabel('Odds', fontdict=self.font1, labelpad=6)
        return ax