class FeatureStability:
    '''可视化特征在训练集和测试集上的分布，可以用来发现不稳定的特征。也可以用来可视化特征在不同类别的特征，
    用来选取重要特征或者删除不重要特征。通过函数发现训练和测试集分布不一致的特征，返回不一致的特征'''

    def __init__(self, threshold=0.05):
        self.pvalue = threshold

    def num_stab_test(self, train, test, feature=None):
        '''Compute the Kolmogorov-Smirnov statistic on 2 samples.
        检验数值特征在训练集和测试集分布是否一致,ks检验，null hypothesis是两个样本取自
        同一分布，当pvalue小于设定阈值，则拒绝原假设，则训练集和测试集的特征样本不是取自同一
        分布。可以考虑是否去除这个特征。'''
        if feature is None:
            _, pvalue = stats.ks_2samp(train, test)
        else:
            _, pvalue = stats.ks_2samp(train[feature], test[feature])
        return pvalue

    def num_stab_tests(self, train, test, features):
        values = [
            self.num_stab_test(train, test, feature) for feature in features
        ]
        mask = [value > self.pvalue for value in values]
        return mask, values

    def get_value_count(self, train, test, feature=None, normalize=True):
        if feature is None:
            count = train.value_counts(normalize=normalize)
            count1 = test.value_counts(normalize=normalize)
        else:
            count = train[feature].value_counts(normalize=normalize)
            count1 = test[feature].value_counts(normalize=normalize)
        index = count.index | count1.index
        if normalize:
            count = count.reindex(index).fillna(1e-3)
            count1 = count1.reindex(index).fillna(1e-3)
        else:
            count = count.reindex(index).fillna(1)
            count1 = count1.reindex(index).fillna(1)
        return count, count1

    def psi(self, train, test, feature=None):  # Population Stability Index
        '''PSI大于0.1则视为不太稳定。越小越稳定,通过PSI来评估特征在训练集和测试集上
        分布的稳定性。'''
        count, count1 = self.get_value_count(train, test, feature)
        res = (count1 - count)*np.log(count1/count)
        return res.sum()

    def psis(self, train, test, features):
        value = [self.psi(train, test, feature) for feature in features]
        res = pd.Series(value, index=features)
        return res

    def cat_stab_test(self, train, test, feature=None):
        '''检验类别特征在训练集和测试集分布是否一致。chi2检验，null hypothesis为分布相互
        对立，所以pvalue小于设定值拒绝原假设，即特征的分布与训练集和测试集有关，即
        特征分布在训练集和测试集不是一致的。'''
        count, count1 = self.get_value_count(train,
                                             test,
                                             feature,
                                             normalize=False)
        data = pd.concat([count, count1], axis=1)
        _, pvalue, *_ = stats.chi2_contingency(data.to_numpy().T,
                                               correction=False)
        return pvalue

    def cat_stab_tests(self, train, test, features):
        values = [
            self.cat_stab_test(train, test, feature) for feature in features
        ]
        mask = [value > self.pvalue for value in values]
        return mask, values

    def get_labels(self, labels=None):
        if labels is None:
            label1, label2 = 'Train', 'Test'
            return label1, label2
        elif isinstance(labels, Iterable) and len(labels) >= 2:
            label1, label2 = labels[0], labels[1]
            return label1, label2
        else:
            raise ValueError('labels is wrong!')

    def plot_train_test_num(self, train, test, features, labels=None):
        '''可视化数值特征在训练集和测试集上的分布。'''
        label1, label2 = self.get_labels(labels)
        if isinstance(features, str):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            fig.suptitle('Distribution of values in {} and {}'.format(
                label1, label2),
                         fontsize=16)
            ax.set_title('Distribution of {}'.format(features))
            sns.distplot(train[features],
                         color="green",
                         kde=True,
                         bins=50,
                         label=label1,
                         ax=ax)
            sns.distplot(test[features],
                         color="blue",
                         kde=True,
                         bins=50,
                         label=label2,
                         ax=ax)
            plt.legend(loc=2)
        elif isinstance(features, Iterable):
            nrows = len(features)//4+1
            fig = plt.figure(figsize=(20, 5*nrows))
            fig.suptitle('Distribution of values in {} and {}'.format(
                label1, label2),
                         fontsize=16,
                         horizontalalignment='right')
            grid = gridspec.GridSpec(nrows, 4)
            for i, each in enumerate(features):
                ax = fig.add_subplot(grid[i])
                sns.distplot(train[each],
                             color="green",
                             kde=True,
                             bins=50,
                             label=label1,
                             ax=ax)
                sns.distplot(test[each],
                             color="blue",
                             kde=True,
                             bins=50,
                             label=label2,
                             ax=ax)
                ax.set_xlabel('')
                plt.legend(loc=2)
                ax.set_title('Distribution of {}'.format(each))
                plt.legend(loc=2)
        else:
            raise TypeError('{} is not right datatype'.format(type(features)))

    def get_melt(self, train, test, feature, labels):
        res = train[feature].value_counts(normalize=True)
        res1 = test[feature].value_counts(normalize=True)
        data = pd.concat([res, res1], axis=1).fillna(0)
        data.columns = labels
        data = data.reset_index()  # index变为column后的name默认为index
        melt = pd.melt(data, id_vars='index')
        return melt

    def plot_train_test_cat(self, train, test, features, labels=None):
        '''可视化类别特征在训练集和测试集上的分布。'''
        label1, label2 = self.get_labels(labels)
        if isinstance(features, str):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            fig.suptitle('Distribution of values in {} and {}'.format(
                label1, label2),
                         fontsize=16)
            ax.set_title('{}'.format(features))
            melt = self.get_melt(train, test, features, [label1, label2])
            sns.barplot(x='index', y='value', data=melt, hue='variable', ax=ax)
            plt.legend(loc=2)
        elif isinstance(features, Iterable):
            nrows = len(features)//4 + 1
            fig = plt.figure(figsize=(20, 5*nrows))
            fig.suptitle('Distribution of values in {} and {}'.format(
                label1, label2),
                         fontsize=16,
                         horizontalalignment='right')
            grid = gridspec.GridSpec(nrows, 4)
            for i, each in enumerate(features):
                ax = fig.add_subplot(grid[i])
                melt = self.get_melt(train, test, each, [label1, label2])
                sns.barplot(x='index',
                            y='value',
                            data=melt,
                            hue='variable',
                            ax=ax)
                ax.set_title('{}'.format(each))
                plt.legend(loc=2)
        else:
            raise TypeError('{} is not right datatype'.format(type(features)))

    def target_split_data(self, data, target):
        '''根据目标特征对训练集进行划分。'''
        mask = data[target] == 0
        train = data.loc[mask, :]
        test = data.loc[~mask, :]
        labels = ['Target=0', 'Target=1']
        return train, test, labels

    def plot_target_feature_cat(self, data, features, target):
        '''可视化类别特征在目标变量上的分布。'''
        train, test, labels = self.target_split_data(data, target)
        self.plot_train_test_cat(train, test, features, labels)

    def plot_target_feature_num(self, data, features, target):
        '''可视化数值特征在目标变量上的分布。'''
        train, test, labels = self.target_split_data(data, target)
        self.plot_train_test_num(train, test, features, labels)
