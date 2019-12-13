import warnings
import numpy as np
from queue import Queue
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def ensure_rng(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

def _hashable(x):
    return tuple(map(float,x))

class UtilityFunction:
    def __init__(self, kind, kappa, xi):
        self.kappa = kappa
        self.xi = xi
        
        if kind not in ['ucb', 'ei', 'poi']:
            err = 'The Utility function {} has not been implemented'.format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind
    
    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean, std = gp.predict(x, return_std=True)
        return mean + kappa*std

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)
    
class TargetSpace:
    def __init__(self, target_func, pbounds, random_state=None):
        self.random_state = ensure_rng(random_state)
        self.target_func = target_func
        self._keys = sorted(pbounds)
        self._bounds = np.array([item[1] for item in sorted(pbounds.items(),
                                  key=lambda x: x[0])], dtype = np.float)

        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))
        self._cache = { }
    def __contains__(self, x):
        return _hashable(x) in self._cache
    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)
    @property
    def empty(self):
        return len(self) == 0
    @property
    def params(self):
        return self._params
    @property
    def target(self):
        return self._target
    @property
    def keys(self):
        return self._keys        
    @property
    def dim(self):
        return len(self._keys)
    @property
    def bounds(self):
        return self._bounds
    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError('Paramters do not match')
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError('Paramters do not match')
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x,dtype=float)
        except TypeError:
            x = self.params_to_array(x)
        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError('Size of array is different than the expected')
        return x
    def register(self, params, target):
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point is not unique')
        self._cache[_hashable(x.ravel())] = target
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params):
        x = self._as_array(params)
        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target
    def random_sample(self):
        data = np.empty(self.dim)
        for col, (lower, upper) in enumerate(self._bounds):
            data[col] = self.random_state.uniform(lower, upper,size=1)
        return data
    
    def max(self):
        try:
            res = {'target': self.target.max(),
                'params': dict(zip(self.keys, self.params[self.target.argmax()]))}
        except ValueError:
            res = {}
        return res

    def res(self):
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [{"target": target, "params": param}
            for target, param in zip(self.target, params)]

def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))# 产生样本点
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")
        if not res.success:
            continue
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

class Queues(Queue):
    def __next__(self):
        if self.empty():
            raise StopIteration
        else:
            return self.get()

class BayesianOptimization:
    def __init__(self, f, pbounds, random_state=None):
        """"""
        self._random_state = ensure_rng(random_state)

        self._space = TargetSpace(f, pbounds, random_state)

        self._gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6,
            normalize_y=True, n_restarts_optimizer=5, random_state=self._random_state,)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        self._space.register(params, target)

    def probe(self, params, lazy=True):
        if lazy:
            self._queue.put(params)
        else:
            self._space.probe(params)

    def suggest(self, utility_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        suggestion = acq_max(ac=utility_function.utility, gp=self._gp, y_max=self._space.target.max(),
            bounds=self._space.bounds,random_state=self._random_state)

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty() and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.put(self._space.random_sample())

    def maximize(self, init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, **gp_params):
        """Mazimize your function"""
        self._queue = Queues(init_points+n_iter)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty() or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

from sklearn.model_selection import cross_val_score
from BayesOpt import BayesianOptimization
from sklearn import clone
class Bayes:
    def __init__(self,estimator=None, params=None):
        self.estimator = estimator
        self.params = params
        self.max_ = None

    def rfc_cv(self, data, targets, **kwarg):
        estimator = clone(self.estimator)
        estimator = estimator.set_params(**kwarg)

        cval = cross_val_score(estimator, data, targets,
                            scoring='neg_log_loss', cv=4)
        return cval.mean( )
    
    def fit(self, data, target):
        def opt_func(**kwarg):
            return self.rfc_cv(data = data, target=target, **kwarg)
        
        optimizer = BayesianOptimization(f=opt_func, pbounds=self.params,
        random_state=1234)
        optimizer.maximize(n_iter=10)
        self.max_ = optimizer.max
        return self.max_