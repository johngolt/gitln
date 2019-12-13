import random
import math
import numpy as np

def _hashable(x):
    return tuple(map(float,x))

def ensure_rng(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

class Mutation:
    pass

class Selection:
    pass

class CrossOver:
    pass

class Population:
    def __init__(self, target_func, pbounds, random_state=None):
        self.random_state = ensure_rng(random_state)
        self.target_func = target_func
        self._keys = sorted(pbounds)
        self._bounds = np.array([item[1] for item in sorted(pbounds.items(),key=lambda x: x[0])],
        dtype = np.float)
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
    pass 


class Genetic:
    pass 