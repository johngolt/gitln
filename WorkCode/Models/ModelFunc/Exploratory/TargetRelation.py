class CalEnt:
    '''比较某字段在类别合并前后的信息熵、基尼值、信息值IV，若合并后信息熵值减小/基尼值减小/信息值IV增大，
    则确认合并类别有助于提高此字段的分类能力，可以考虑合并类别。'''
    def _entropy(self, group):  # 计算信息熵
        return -group*np.log2(group+1e-5)

    def _gini(self, group):  # 计算基尼系数
        return group*(1-group)

    def cal(self, data, feature, target, func=None):
        '''计算使用的通用函数'''
        temp1 = data[feature].value_counts(normalize=True)
        temp = pd.crosstab(data[feature], data[target], normalize='index')
        enti = func(temp)
        return (temp1*enti.sum(axis=1)).sum()

    def calculates(self, data, features, target, func):
        '''通用函数，用来处理多个特征的计算'''
        if isinstance(features, str):
            return func(data, features, target)
        elif isinstance(features, Iterable):
            data = [func(data, feature, target) for feature in features]
            result = pd.Series(data, index=features)
            return result
        else:
            raise TypeError('Features is not right!')

    def entropy(self, data, feature, target):  # 计算条件信息熵
        return self.cal(data, feature, target, self._entropy)

    def entropys(self, data, features, target):
        '''计算条件信息熵的接口，通过条件信息熵可以评价特征的重要程度。'''
        return self.calculates(data, features, target, self.entropy)

    def gini(self, data, feature, target):  # 计算条件基尼系数
        return self.cal(data, feature, target, self._gini)

    def ginis(self, data, features, target):
        '''计算条件信息系数的接口，通过条件信息系数可以评价特征的重要程度。'''
        return self.calculates(data, features, target, self.entropy)

    def woe(self, data, feature, target):
        '''计算woe值,可以用于类别特征的编码'''
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = np.log((temp.iloc[:, 0]+1e-5)/(temp.iloc[:, 1]+1e-5))
        return woei

    def iv(self, data, feature, target):
        '''计算IV值，通过IV值可以进行特征选择，一般要求IV值大于0.02'''
        temp = pd.crosstab(data[feature], data[target], normalize='columns')
        woei = self.woe(data, feature, target)
        iv = (temp.iloc[:, 0] - temp.iloc[:, 1])*woei
        return iv.sum()

    def ivs(self, data, features, target):
        return self.calculates(data, features, target, self.iv)

    def woes(self, data, features, target):
        if isinstance(features, str):
            return self.woe(data, features, target)
        elif isinstance(features, Iterable):
            return{each: self.woe(data, each, target) for each in features}
        else:
            raise TypeError('Feature is not right data type')
