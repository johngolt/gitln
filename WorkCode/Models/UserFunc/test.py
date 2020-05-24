import sklearn.model_selection as ms
from sklearn import clone
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class HyOptimize:
    def __init__(self,
                 estimator=None,
                 params=None,
                 cv=5,
                 grid=True,
                 stratified=False):
        if stratified:
            self.cv = ms.StratifiedKFold(cv, shuffle=True)
        else:
            self.cv = cv
        self.grid = grid
        self.estimator = estimator
        self.params = params
        self.res = {}
        self.scoring = 'roc_auc'

    def CoordinateAscent(self, X, y, estimator, params, cv, scoring):
        name = estimator.__class__.__name__
        res, i, means, stds = {}, 0, [], []
        temp = {
            key: value
            for key, value in estimator.get_params().items() if key in params
        }
        while i <= 20 and res != temp:
            res, temp = temp, {}
            stepmean, stepstd = [], []
            for each in params:
                grid = ms.GridSearchCV(estimator,
                                       param_grid={each: params[each]},
                                       scoring=scoring,
                                       cv=cv,
                                       iid=True,
                                       n_jobs=-1)
                grid.fit(X, y)
                mean = grid.cv_results_['mean_test_score'].tolist()
                std = grid.cv_results_['std_test_score'].tolist()
                stepmean.extend(mean)
                stepstd.extend(std)
                temp.update(grid.best_params_)
                estimator = clone(grid.best_estimator_)
            means.append(stepmean)
            stds.append(stepstd)
            i += 1
        self.estimator = clone(grid.best_estimator_)
        self.res[name + '_mean'] = np.array(means[-1])
        self.res[name + '_std'] = np.array(stds[-1])
        return grid.best_score_, temp

    def GridSearch(self, X, y, estimator, params, cv, scoring):
        name = estimator.__class__.__name__
        grid = ms.GridSearchCV(estimator,
                               param_grid=params,
                               scoring=scoring,
                               cv=cv,
                               iid=True,
                               n_jobs=-1)
        grid.fit(X, y)
        self.res[name +
                 '_mean'] = grid.cv_results_['mean_test_score']  # 保存搜索过程中的结果
        self.res[name + '_std'] = grid.cv_results_['std_test_score']
        self.estimator = clone(grid.best_estimator_)
        return grid.best_score_, grid.best_params_

    def hyparameter(self,
                    X,
                    y,
                    estimator=None,
                    params=None,
                    cv=None,
                    scoring=None,
                    grid=None):
        if estimator is None:
            estimator = self.estimator
        if params is None:
            params = self.params
        if cv is None:
            cv = self.cv
        if scoring is None:
            scoring = self.scoring
        if grid is None:
            grid = self.grid
        if grid:  # 采用网格搜索
            return self.GridSearch(X, y, estimator, params, cv, scoring)
        else:  # 采用梯度上升算法进行搜索
            return self.CoordinateAscent(X, y, estimator, params, cv, scoring)

    def plot_optimizer(self, estimator):
        name = estimator.__class__.__name__
        mean = self.res[name + '_mean']
        std = self.res[name + '_std']
        x = range(len(mean))
        ax = plt.subplot()
        color = ("#2492ff", "#ff9124")
        ax.plot(x, mean, color=color[0], lw=2, alpha=0.3)
        ax.fill_between(x, mean + std, mean - std, alpha=0.1, color=color[1])

    def fit(self, X, y, search=False):
        if search:
            _ = self.hyparameter(X, y)
        self.estimator = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_auc_ks(self, y, y_pred):
        fpr, tpr, _ = roc_curve(y, y_pred, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        ks = (tpr - fpr).max()
        return roc_auc, ks

    def plot_auc_ks(self, y, y_pred):
        fpr, tpr, threshold = roc_curve(y, y_pred, drop_intermediate=False)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        ax1.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=2,
                 label='ROC Curve(area=%0.2f)' % auc(fpr, tpr))
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel('False Positive Rate')
        ax1.set(xlabel='False Positve Rate',
                ylabel='True Positive Rate',
                title='ROC')
        ax1.legend(loc='lower right')
        index = (tpr - fpr).argmax()
        ax2.plot(threshold, 1 - tpr, label='tpr')
        ax2.plot(threshold, 1 - fpr, label='fpr')
        ax2.plot(threshold,
                 tpr - fpr,
                 label='KS %0.2f' % (tpr - fpr)[index],
                 linestyle='--')
        ax2.vlines(threshold[index], 1 - fpr[index], 1 - tpr[index])
        ax2.set(xlabel='score')
        ax2.set_xlim(0, 1)
        ax2.set_title('KS Curve')
        ax2.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()


class PlotCV:
    def plot_cv_curve(self, results, x=None):
        if x is None:
            x = range(results[0].shape[0])
        colors = cycle([("#2492ff", "#ff9124"), ('blue', 'green')])
        ax = plt.subplot()
        for color, (key, result) in zip(colors, results.items()):
            mean, std = result.mean(axis=1), result.std(axis=1)
            ax.plot(x, mean, color=color[0], lw=2, alpha=0.5, label=key)
            ax.fill_between(x,
                            mean + std,
                            mean - std,
                            alpha=0.1,
                            color=color[1])
        return ax

    def plot_learning_curve(self,
                            estimator1,
                            X,
                            y,
                            cv=5,
                            train_sizes=np.linspace(.1, 1.0, 5)):
        name = estimator1.__class__.__name__
        train_sizes, trains, tests = ms.learning_curve(estimator1,
                                                       X,
                                                       y,
                                                       cv=cv,
                                                       n_jobs=-1,
                                                       train_sizes=train_sizes)
        ax = self.plot_cv_curve({
            'train': trains,
            'test': tests
        },
                                x=train_sizes)
        ax.set_title("{} Learning Curve".format(name), fontsize=14)
        ax.set_xlabel('Training size')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")

    def plot_feature_importance(self, result):
        ax = self.plot_cv_curve({'feature importance': result})
        mean = result.mean(axis=1)
        ax.scatter(mean.argmax(),
                   mean.max(),
                   color='black',
                   s=50,
                   edgecolors='purple')
        ax.vlines(mean.argmax(), 0, mean.max())
        ax.text(mean.argmax(), mean.max() * 1.01, '{:.2f}'.format(mean.max()))
        ax.set_ylim(mean.min() * 0.9, mean.max() * 1.1)


class AveragingModels(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack(
            [model.predict(X) for model in self.models_])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                   axis=1,
                                   arr=predictions)

    def predict_proba(self, X):
        predictions = np.array(
            [model.predict_proba(X) for model in self.models_])
        return np.mean(predictions, axis=0)


def plot_pr_curve(y_true, y_pred):
    AP = metrics.average_precision_score(y_true, y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.step(recall, precision, color='c', where='post', alpha=0.5)
    ax.fill_between(recall, precision, color='b', alpha=0.2)
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1.), ylim=(0, 1.))
    ax.set_title('Average Precision Score:{:.2f}'.format(AP), fontsize=16)
