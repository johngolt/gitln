import random
import numpy as np
import matplotlib.pyplot as plt


def mh(q, p, m, n):
    # randomize a number
    x = random.uniform(0.1, 1)
    for t in range(0, m+n):
        x_sample = q.sample(x)
        try:
            accept_prob = min(1, p.prob(x_sample)*q.prob(x_sample, x)/(p.prob(x)*q.prob(x, x_sample)))
        except:
            accept_prob = 0

        u = random.uniform(0, 1)

        if u < accept_prob:
            x = x_sample

        if t >= m:
            yield x


class Exponential(object):
    def __init__(self, scale):
        self.scale = scale
        self.lam = 1.0 / scale

    def prob(self, x):
        if x <= 0:
            raise Exception("The sample shouldn't be less than zero")

        result = self.lam * np.exp(-x * self.lam)
        return result

    def sample(self, num):
        sample = np.random.exponential(self.scale, num)
        return sample


# 假设我们的目标概率密度函数p1(x)是指数概率密度函数
scale = 5
p1 = Exponential(scale)


class Norm():
    def __init__(self, mean, std):
        self.mean = mean
        self.sigma = std

    def prob(self, x):
        return np.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2.0)) * 1.0 / (np.sqrt(2 * np.pi) * self.sigma)

    def sample(self, num):
        sample = np.random.normal(self.mean, self.sigma, size=num)
        return sample

# 假设我们的目标概率密度函数p1(x)是均值方差分别为3,2的正态分布
p2 = Norm(3, 2)


class Transition():
    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, cur_mean):
        cur_sample = np.random.normal(cur_mean, scale=self.sigma, size=1)[0]
        return cur_sample

    def prob(self, mean, x):
        return np.exp(-(x-mean)**2/(2*self.sigma**2.0)) * 1.0/(np.sqrt(2 * np.pi)*self.sigma)


# 假设我们的转移核方差为10的正态分布
q = Transition(10)

m = 100
n = 100000 # 采样个数

simulate_samples_p1 = [li for li in mh(q, p1, m, n)]

plt.subplot(2,2,1)
plt.hist(simulate_samples_p1, 100)
plt.title("Simulated X ~ Exponential(1/5)")

samples = p1.sample(n)
plt.subplot(2,2,2)
plt.hist(samples, 100)
plt.title("True X ~ Exponential(1/5)")

simulate_samples_p2 = [li for li in mh(q, p2, m, n)]
plt.subplot(2,2,3)
plt.hist(simulate_samples_p2, 50)
plt.title("Simulated X ~ N(3,2)")


samples = p2.sample(n)
plt.subplot(2,2,4)
plt.hist(samples, 50)
plt.title("True X ~ N(3,2)")

plt.suptitle("Transition Kernel N(0,10)simulation results")
plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Transition():
    def __init__(self, mean, cov):
        self.mean = mean
        self.sigmas = []
        for i in range(K):
            self.sigmas.append(np.sqrt(cov[i][i]))
        self.rho = cov[0][1]/(self.sigmas[0] * self.sigmas[1])

    def sample(self, id1, id2_list, x2_list):
        id2 = id2_list[0]  # only consider two dimension
        x2 = x2_list[0]  # only consider two dimension
        cur_mean = self.mean[id1] + self.rho*self.sigmas[id1]/self.sigmas[id2] * (x2-self.mean[id2])
        cur_sigma = (1-self.rho**2) * self.sigmas[id1]**2
        return np.random.normal(cur_mean, scale=cur_sigma, size=1)[0]


def gibbs(p, m, n):
    # randomize a number
    x = np.random.rand(K)
    for t in range(0, m+n):
        for j in range(K):
            total_indexes = list(range(K))
            total_indexes.remove(j)
            left_x = x[total_indexes]
            x[j] = p.sample(j, total_indexes, left_x)

        if t >= m:
            yield x


mean = [5, 8]
cov = [[1, 0.5], [0.5, 1]]
K = len(mean)
q = Transition(mean, cov)
m = 100
n = 1000

gib = gibbs(q, m, n)

simulated_samples = []

x_samples = []
y_samples = []
for li in gib:
    x_samples.append(li[0])
    y_samples.append(li[1])


fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')

hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=100, range=[[0,10],[0,16]])
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax = fig.add_subplot(132)
ax.hist(x_samples, bins=50)
ax.set_title("Simulated on dim1")

ax = fig.add_subplot(133)
ax.hist(y_samples, bins=50)
ax.set_title("Simulated on dim2")
plt.show()