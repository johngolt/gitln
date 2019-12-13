class SequenceIterator:
    ''' 迭代器类, 利用生成器来迭代器功能'''
    def __init__(self, sequence):
        self._seq = sequence
        self._k = -1

    def __next__(self):
        self._k+=1
        if self._k<len(self._seq):
            return self._seq[self._k]
        else:
            raise StopIteration()

    def __iter__(self):
        return self

class Range:
    ''' A class that mimic's the built-in range class'''
    def __init__(self, start, stop = None, step = 1):
        ''' Initialize the Range instance.  Semantics is similar to built-in range class'''
        if step==0:
            raise ValueError('step cannot be 0')
        if stop is None:
            start, stop = 0, start
        # calculate the effective length once
        self._length = max(0, (stop-step+step-1)//step)
        self._start = start
        self._step = step
    def __len__(self):
        return self._length
    def __getitem__(self, k):
        if k<0:
            k+=len(self)
        if not 0<=k<self._length:
            raise IndexError('index out of range')
        return self._start + k*self._step

# 类的继承
class Progression:
    def __init__(self, start):
        self._current = start

    def _advance(self):
        self._current+=1

    def __next__(self):
        if self._current is None:
            raise StopIteration()
        else:
            answer = self._current
            self._advance()
            return answer
    def __iter__(self):
        return self
    def print_progression(self, n):
        print(' '.join(str(next(self)) for j in range(n)))

class ArithmeticProgression(Progression):
    def __init__(self, increment = 1, start = 0):
        super().__init__(start)
        self._increment = increment

    def _advance(self):
        self._current+=self._increment

class GeometricProgression(Progression):
    def __init__(self, base = 2, start = 1):
        super().__init__(start)
        self._base = base
    def _advance(self):
        self._current*=self._base

class FibonacciProgression(Progression):
    def __init__(self, first = 0, second = 1):
        super().__init__(first)
        self._prev = second - first

    def _advance(self):
        self._prev, self._current = self._current, self._current+self._prev


# 定义抽象基类, 抽象基类不可以实例化
from abc import ABCMeta, abstractmethod

class Sequence(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        '''在 base类中不提供具体实现, 但在任何具体的子类来实现此方法'''
        pass
    @abstractmethod
    def __getitem__(self, item):
        pass

    def __contains__(self, item):
        for j in range(len(self)):
            if self[j] == item:
                return True
        return False
    def index(self, val):
        for j in range(len(self)):
            if self[j] == val:
                return j
        raise ValueError('value not in sequence')

    def count(self, val):
        k = 0
        for j in range(len(self)):
            if self[j] == val:
                k+=1
        return k
