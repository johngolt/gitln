##### 2.1. Gaussian mixture models

The GaussianMixture comes with different options to constrain the covariance of the difference classes estimated: spherical, diagonal, tied or full covariance.
The BIC criterion can be used to select the number of components in a Gaussian Mixture in an efficient way. In theory, it recovers the true number of components only in the asymptotic regime. Note that using a Variational Bayesian Gaussian mixture avoids the specification of the number of components for a Gaussian mixture model.

######  EM算法

令$\mathbf{Y}$表示观测随机变量，$\mathbb{Y}=\left\{y_{1}, y_{2}, \cdots, y_{N}\right\}$表示对应的数据序列；令$\mathbf{Z}$表示隐随机变量，$\mathbb{Z}=\left\{z_{1}, z_{2}, \cdots, z_{N}\right\}$表示对应的数据序列。$\mathbb{Y}$和$\mathbb{Z}$连在一起称作完全数据，观测数据$\mathbb{Y}$又称作不完全数据。假设给定观测随机变量$\mathbf{Y}$，其概率分布为$P(Y ; \theta)$，其中$\theta$是需要估计的模型参数，则不完全数据$\mathbb{Y}$的似然函数是$P(\mathbb{Y} ; \theta)$， 对数似然函数为$L(\theta)=\log P(\mathbb{Y} ; \theta)$。

假定$\mathbf{Y}$和$\mathbf{Z}$的联合概率分布是$P(Y, Z ; \theta)$，完全数据的对数似然函数是$\log P(\mathbb{Y}, \mathbb{Z} ; \theta)$，则根据每次观测之间相互独立，有：
$$
\begin{array}{c}{\log P(\mathbb{Y} ; \theta)=\sum_{i} \log P\left(Y=y_{i} ; \theta\right)} \\ {\log P(\mathbb{Y}, \mathbb{Z} ; \theta)=\sum_{i} \log P\left(Y=y_{i}, Z=z_{i} ; \theta\right)}\end{array}
$$
由于$\mathbb{Y}$发生，根据最大似然估计，则需要求解对数似然函数：
$$
\begin{array}{c}{L(\theta)=\log P(\mathbb{Y} ; \theta)=\sum_{i=1} \log P\left(Y=y_{i} ; \theta\right)=\sum_{i=1} \log \sum_{Z} P} \\ {=\sum_{i=1} \log \left[\sum_{Z} P\left(Y=y_{i} | Z ; \theta\right) P(Z ; \theta)\right]}\end{array}
$$
的极大值。其中$\sum_{Z} P\left(Y=y_{i}, Z ; \theta\right)$表示对所有可能的$\mathbf{Z}$求和，因为边缘分布$P(Y)=\sum_{Z} P(Y, Z)$。

`EM` 算法通过迭代逐步近似极大化$L(\theta)$。

假设在第$i$次迭代后，$\theta$的估计值为：$\theta^i$。则希望$\theta$新的估计值能够使得$L(\theta)$增加。为此考虑两者的差：$L(\theta)-L\left(\theta^{i}\right)=\log P(\mathbb{Y} ; \theta)-\log P\left(\mathbb{Y} ; \theta^{i}\right)$

`Jensen`不等式：如果$f$是凸函数，$x$为随机变量，则有：$\mathbb{E}[f(x)] \leq f(\mathbb{E}[x])$。

考虑到条件概率的性质，则有$\sum_{Z} P(Z | Y ; \theta)=1$。因此有
$$
\begin{array}{1}{L(\theta)-L\left(\theta^{i}\right)=\sum_{j} \log \sum_{Z} P\left(Y=y_{j}, Z ; \theta\right)-\sum_{j} \log P\left(Y=y_{j} ; \theta^{i}\right)} \\ {=\sum_{j}\left[\log \sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \frac{P\left(Y=y_{j}, Z ; \theta\right)}{P\left(Z | Y=y_{j} ; \theta^{i}\right)}-\log P\left(Y=y_{j} ; \theta^{i}\right)\right]}\\
\geq \sum_{j}\left[\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log \frac{P\left(Y=y_{j}, Z ; \theta\right)}{P\left(Z | Y=y_{j} ; \theta^{i}\right)}-\log P\left(Y=y_{j} ; \theta^{i}\right)\right]\\
=\sum_{j} \left[\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log \frac{P\left(Y=y_{j} | Z ; \theta\right) P(Z ; \theta)}{P\left(Z | Y=y_{j} ; \theta^{i}\right)}\right.-\log P\left(Y=y_{j} ; \theta^{i}\right) \times \sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) ]\\
=\sum_{j}\left[\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log \frac{P\left(Y=y_{j} | Z ; \theta\right) P(Z ; \theta)}{P\left(Z | Y=y_{j} ; \theta^{i}\right) P\left(Y=y_{j} ; \theta^{i}\right)}\right]\\
\text{令}:\\
B\left(\theta, \theta^{i}\right)=L\left(\theta^{i}\right)+\sum_{j}\left[\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log \frac{P\left(Y=y_{j} | Z ; \theta\right) P(Z ; \theta)}{P\left(Z | Y=y_{j} ; \theta^{i}\right) P\left(Y=y_{j} ; \theta^{i}\right)}\right]
\end{array}
$$
任何可以使得$B(\theta, \theta^{i})$增大的$\theta$，也可以使$L(\theta)$增大。为了使得$L(\theta)$尽可能增大，则选择使得$B(\theta, \theta^{i})$取极大值的$\theta$：$\theta^{i+1}=\arg \max _{\theta} B\left(\theta, \theta^{i}\right)$。
$$
\begin{array}{l}\theta^{i+1}=\arg \max _{\theta} B\left(\theta, \theta^{i}\right)\\{=\arg \max _{\theta} \sum_{j}\left(\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log P\left(Y=y_{j} | Z ; \theta\right) P(Z ; \theta)\right)} \\ {=\arg \max _{\theta} \sum_{j}\left(\sum_{Z} P\left(Z | Y=y_{j} ; \theta^{i}\right) \log P\left(Y=y_{j}, Z ; \theta\right)\right)}\end{array}
$$

###### 原理

假设混合高斯模型由K个高斯模型组成，则GMM的概率密度函数如下：
$$
p(x) = \sum_{k=1}^{K}p(k)p(x|k) = \sum_{k=1}^{K}\pi_kN(x|u_k,\Sigma_k)
$$
先引入一个隐变量$\gamma$。它是一个K维二值随机变量，在它的K维取值中只有某个特定的元素$\gamma_k$的取值为1，其它元素的取值为0。实际上，隐变量描述的就是：每一次采样，选择第k个高斯模型的概率，故有：$p(\gamma_k=1)=\pi_k$. 当给定了$\gamma$的一个特定的值之后，可以得到样本y的条件分布是一个高斯分布，满足：$p(y|\gamma_k=1) = N(y|u_k,\Sigma_k)$. 而实际上，每个样本到底是从这K个高斯模型中哪个模型进行采样的，是都有可能的。故样本y的概率为：$p(y)=\Sigma_{\gamma}p(\gamma)p(y|\gamma)$. 样本集Y的联合概率为：
$$
L(\mu,\Sigma,\pi) = L(y_1,y_2,\cdots,y_N;\mu,\Sigma,\pi) = \prod_{n=1}^{N}\sum_{k=1}^{K}\pi_kN(y_n|\mu_k,\Sigma_k) 
$$
对数似然函数表示为：
$$
ln(L(\mu,\Sigma,\pi)) = \sum_{n=1}^{N}\ln(\sum_{k=1}^{K}\pi_kN(y_n|\mu_k,\Sigma_k))
$$


##### 2.2. Manifold learning

###### 多维缩放：`MDS`

多维缩放要求原始空间中样本之间的距离在低维空间中得到保持。

![](D:/学习/MarkDown/picture/1/80.png)

###### 等度量映射

等度量映射的基本观点是：低维流形嵌入到高维空间后，直接在高维空间中计算直线距离具有误导性。因为在高维空间中的直线距离在低维嵌入流形上是不可达的。利用流形在局部上与欧氏空间同胚这个性质，对每个点基于欧氏距离找出它在低维流形上的近邻点， 然后就能建立一个近邻连接图。图中近邻点之间存在链接。图中非近邻点之间不存在链接。于是计算两点之间测地线距离的问题转变为计算近邻连接图上两点之间的最短路径问题。在得到任意两点的距离之后，就可以通过`MDS`算法来获得样本点在低维空间中的坐标。

![](D:/学习/MarkDown/picture/1/81.png)

###### 局部线性嵌入：`LLE`

假定样本点$\vec{\mathbf{x}}_i$的坐标能够通过它的邻域样本$\vec{\mathbf{x}}_{j}, \vec{\mathbf{x}}_{k}, \vec{\mathbf{x}}_{l}$进行线性组合而重构出来，即：$\vec{\mathbf{x}}_{i}=w_{i, j} \vec{\mathbf{x}}_{j}+w_{i, k} \vec{\mathbf{x}}_{k}+w_{i, l} \vec{\mathbf{x}}_{l}$。`LLE`算法希望这种关系在低维空间中得到保持。

`LLE`首先为每个样本$\vec{\mathbf{x}}$找到其近邻点下标集合$\mathbf{Q}_i$， 然后计算基于$\mathbf{Q}_i$中的样本点对$\vec{\mathbf{x}}$进行线性重构的系数$\overrightarrow{\mathbf{w}}_{i}$。定义样本集重构误差为：$e r r=\sum_{i=1}^{N}\left\|\overrightarrow{\mathbf{x}}_{i}-\sum_{j \in \mathbb{Q}_{i}} w_{i, j} \overrightarrow{\mathbf{x}}_{j}\right\|_{2}^{2}$。目标是样本集重构误差最小，即：$\min _{\overrightarrow{\mathbf{w}}_{1}, \overrightarrow{\mathbf{w}}_{2}, \cdots, \overrightarrow{\mathbf{w}}_{N}} \sum_{i=1}^{N}\left\|\overrightarrow{\mathbf{x}}_{i}-\sum_{j \in \mathbb{Q}_{i}} w_{i, j} \overrightarrow{\mathbf{x}}_{j}\right\|_{2}^{2}$。这样的解有无数个，对权重增加约束，进行归一化处理。即：$\sum_{j \in \mathbb{Q}_{i}} w_{i, j}=1, i=1,2, \cdots, N$。现在就是求解最优化问题：
$$
\begin{array}{l}{\min _{\overrightarrow{\mathbf{w}}_{1}, \overrightarrow{\mathbf{w}}_{2}, \cdots, \overrightarrow{\mathbf{w}}_{N}} \sum_{i=1}^{N}\left\|\overrightarrow{\mathbf{x}}_{i}-\sum_{j \in \mathbb{Q}_{i}} w_{i, j} \overrightarrow{\mathbf{x}}_{j}\right\|_{2}^{2}} \\ {\text {s.t. } \sum_{j \in \mathbb{Q}_{i}} w_{i, j}=1, i=1,2, \cdots, N}\end{array}
$$
该最优化问题有解析解。令$C_{j, k}=\left(\overrightarrow{\mathbf{x}}_{i}-\overrightarrow{\mathbf{x}}_{j}\right)^{T}\left(\overrightarrow{\mathbf{x}}_{i}-\overrightarrow{\mathbf{x}}_{k}\right)$，则可以解出：$w_{i, j}=\frac{\sum_{k \in \mathbb{Q}_{i}} C_{j, k}^{-1}}{\sum_{l, s \in \mathbb{Q}_{i}} C_{l, s}^{-1}}, j \in \mathbb{Q}_{i}$。`LLE` 中出现了两个重构误差。第一个重构误差：为了在原始空间中求解线性重构的系数$\vec{\mathbf{w}}_i$。目标是：基于$\mathbb{Q}_i$中的样本点对$\vec{\mathbf{x}}_i$进行线性重构，使得重构误差最小。第二个重构误差：为了求解样本集在低维空间中的表示$\mathbf{Z}$。目标是：基于线性重构的系数$\vec{\mathbf{w}}_i$，将$\mathbb{Q}_i$中的样本点对$\vec{\mathbf{z}}_i$进行线性重构，使得重构误差最小。

求出了线性重构的系数$\vec{\mathbf{w}}_i$之后， `LLE`在低维空间中保持$\vec{\mathbf{w}}_i$不变。设$\vec{\mathbf{x}}_i$对应的低维坐标$\vec{\mathbf{z}}_i$，已知线性重构的系数  ，定义样本集在低维空间中重构误差为：$e r r^{\prime}=\sum_{i=1}^{N}\left\|\overrightarrow{\mathbf{z}}_{i}-\sum_{j \in \mathbb{Q}_{i}} w_{i, j} \overrightarrow{\mathbf{z}}_{j}\right\|_{2}^{2}$。现在的问题是要求出$\vec{\mathbf{z}}_i$，从而使得上式最小。即求解：$\min _{\overrightarrow{\mathbf{z}}_{1}, \overrightarrow{\mathbf{z}}_{1}, \cdots, \overrightarrow{\mathbf{z}}_{N}} \sum_{i=1}^{N}\left\|\overrightarrow{\mathbf{z}}_{i}-\sum_{j \in \mathbb{Q}_{i}} w_{i, j} \overrightarrow{\mathbf{z}}_{j}\right\|_{2}^{2}$。令$\mathbf{Z}=\left(\overrightarrow{\mathbf{z}}_{1}^{T}, \overrightarrow{\mathbf{z}}_{2}^{T}, \cdots, \overrightarrow{\mathbf{z}}_{N}^{T}\right)^{T} \in \mathbb{R}^{N \times n^{\prime}}$，其中  为低维空间的维数$n^{\prime}$。定义$\mathbf{M}=(\mathbf{I}-\mathbf{W})^{T}(\mathbf{I}-\mathbf{W})$，于是最优化问题可重写为：$\min _{\mathbf{Z}} \operatorname{tr}\left(\mathbf{Z}^{T} \mathbf{M} \mathbf{Z}\right)$。

该最优化问题有无数个解。添加约束$\mathbf{Z}^{T} \mathbf{Z}=\mathbf{I}_{n^{\prime} \times n^{\prime}}$，于是最优化问题为：
$$
\begin{array}{cl}{\min _{\mathbf{Z}}} & {\operatorname{tr}\left(\mathbf{Z}^{T} \mathbf{M} \mathbf{Z}\right)} \\ {\text {s.t.}} & {\mathbf{Z}^{T} \mathbf{Z}=\mathbf{I}_{n^{\prime} \times n^{\prime}}}\end{array}
$$
该最优化问题可以通过特征值分解求解选取$\mathbf{M}$最小的$n^{\prime}$个特征值对应的特征向量组成的矩阵即为$\mathbf{Z}$。

###### $\text{t-SNE}$

`SNE` 的基本思想：如果两个样本在高维相似，则它们在低维也相似。`SNE` 主要包含两步：构建样本在高维的概率分布。在低维空间里重构这些样本的概率分布，使得这两个概率分布之间尽可能相似。

在数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$中，给定一个样本$\overrightarrow{\mathbf{x}}_{i}$，然后计算$\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{i-1}, \overrightarrow{\mathbf{x}}_{i+1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$是$\overrightarrow{\mathbf{x}}_{i}$的邻居的概率。`SNE` 假设：如果$\overrightarrow{\mathbf{x}}_{j}$与$\overrightarrow{\mathbf{x}}_{i}$越相似，则$\overrightarrow{\mathbf{x}}_{j}$是$\overrightarrow{\mathbf{x}}_{i}$的邻居的概率越大。相似度通常采用欧几里得距离来衡量，两个样本距离越近则它们越相似。概率$p\left(\overrightarrow{\mathbf{x}}_{j} | \overrightarrow{\mathbf{x}}_{i}\right)$通常采用指数的形式：$p\left(\overrightarrow{\mathbf{x}}_{j} | \overrightarrow{\mathbf{x}}_{i}\right) \propto \exp \left(-\left\|\overrightarrow{\mathbf{x}}_{j}-\overrightarrow{\mathbf{x}}_{i}\right\|^{2} /\left(2 \sigma_{i}^{2}\right)\right)$对$j=1,2, \cdots, N, j \neq i$进行归一化有：
$$
p\left(\overrightarrow{\mathbf{x}}_{j} | \overrightarrow{\mathbf{x}}_{i}\right)=\frac{\exp \left(-\left\|\overrightarrow{\mathbf{x}}_{j}-\overrightarrow{\mathbf{x}}_{i}\right\|^{2} /\left(2 \sigma_{i}^{2}\right)\right)}{\sum_{k \neq i} \exp \left(-\left\|\overrightarrow{\mathbf{x}}_{k}-\overrightarrow{\mathbf{x}}_{i}\right\|^{2} /\left(2 \sigma_{i}^{2}\right)\right)}
$$
其中$\sigma_i$是与$\overrightarrow{\mathbf{x}}_{i}$相关的、待求得参数，它用于对距离进行归一化。定义$p_{j | i}=p\left(\overrightarrow{\mathbf{x}}_{j} | \overrightarrow{\mathbf{x}}_{i}\right)$。由于挑选时$\overrightarrow{\mathbf{x}}_{j}$排除了$\overrightarrow{\mathbf{x}}_{i}$，因此有$p_{i | i}=0$。定义概率分布$P_{i}=\left(p_{1 | i}, \cdots, p_{N | i}\right)$，它刻画了所有其它样本是$\overrightarrow{\mathbf{x}}_{i}$的邻居的概率分布。

假设经过降维，样本$\overrightarrow{\mathbf{x}}_{i} \in \mathbb{R}^{n}$在低维空间的表示为$\overrightarrow{\mathbf{z}}_{i} \in \mathbb{R}^{d}$，其中$d \leq n$。定义：
$$
q_{j | i}=q\left(\overrightarrow{\mathbf{z}}_{j} | \overrightarrow{\mathbf{z}}_{i}\right)=\frac{\exp \left(-\left\|\overrightarrow{\mathbf{z}}_{j}-\overrightarrow{\mathbf{z}}_{i}\right\|^{2}\right)}{\sum_{k \neq i} \exp \left(-\left\|\overrightarrow{\mathbf{z}}_{k}-\overrightarrow{\mathbf{z}}_{i}\right\|^{2}\right)}
$$
其中$q_{j|i}$表示给定一个样本$\vec{\mathbf{z}}_i$，然后计算$\left\{\overrightarrow{\mathbf{z}}_{1}, \cdots, \overrightarrow{\mathbf{z}}_{i-1}, \overrightarrow{\mathbf{z}}_{i+1}, \cdots, \overrightarrow{\mathbf{z}}_{N}\right\}$是$\vec{\mathbf{z}}_j$的邻居的概率。这里选择$\sigma^2=\frac{1}{2}$为固定值。同样地，有$q_{i|i}=0$。定义概率分布$Q_{i}=\left(q_{1 i}, \cdots, q_{N | i}\right)$，它刻画了所有其它样本是$\vec{\mathbf{z}}_i$的邻居的概率分布。

对于样本$\vec{\mathbf{x}}_i$，如果降维的效果比较好，则有$p_{j | i}=q_{j i}, i=1,2, \cdots, N$。即：降维前后不改变$\vec{\mathbf{x}}_i$周围的样本分布。对于，定义$\vec{\mathbf{x}}_i$其损失函数为分布$\mathbf{P}_I$和$\mathbf{Q}_I$的距离，通过 `KL` 散度来度量。对于全体数据集$\mathbf{D}$，整体损失函数为：
$$
\mathcal{L}=\sum_{i=1}^{N} K L\left(P_{i} \| Q_{i}\right)=\sum_{i=1}^{N} \sum_{j=1}^{N} p_{j | i} \log \frac{p_{j | i}}{q_{j | i}}
$$
记$y_{i, j}=-\left\|\overrightarrow{\mathbf{z}}_{j}-\overrightarrow{\mathbf{z}}_{i}\right\|^{2}$，则有$q_{j | i}=\frac{\exp \left(y_{i, j}\right)}{\sum_{k \neq i} \exp \left(y_{i, k}\right)}$。
$$
\begin{array}{1}
\nabla_{y_{i, j}}\left(\sum_{j=1}^{N} p_{j | i} \log q_{j | \dot{s}}\right)=p_{j | i}-q_{j | i}\\
\begin{array}{c}{\nabla_{\overrightarrow{\mathbf{z}}_{i}}\left(\sum_{j=1}^{N} p_{j | i} \log q_{j | i}\right)=\nabla_{y_{i, j}}\left(\sum_{j=1}^{N}-p_{j | i} \log q_{j | i}\right) \times \nabla_{\overrightarrow{\mathbf{z}}_{i}} y_{i, j}} \\ {=-2\left(p_{j | i}-q_{j | i}\right) \times\left(\overrightarrow{\mathbf{z}}_{i}-\overrightarrow{\mathbf{z}}_{j}\right)}\end{array}\\
\begin{array}{c}{\nabla_{\overrightarrow{\mathbf{a}}_{j}}\left(\sum_{i=1}^{N} p_{j | i} \log q_{j | i}\right)=\nabla_{y_{i, j}}\left(\sum_{i=1}^{N}-p_{j | i} \log q_{j | i}\right) \times \nabla_{\overrightarrow{\mathbf{z}}} y_{i, j}} \\ {=-2\left(p_{j | i}-q_{j | i}\right) \times\left(\overrightarrow{\mathbf{z}}_{j}-\overrightarrow{\mathbf{z}}_{i}\right)}\end{array}
\end{array}
$$
`t-SNE` 通过采用不同的分布来解决拥挤问题：在高维空间下使用高斯分布将距离转换为概率分布。在低维空间下使用 `t` 分布将距离转换为概率分布。`t-SNE` 使用自由度为`1` 的 `t` 分布。此时有：$q_{i, j}=\frac{\left(1+\left\|\overrightarrow{\mathbf{z}}_{i}-\overrightarrow{\mathbf{z}}_{j}\right\|^{2}\right)^{-1}}{\sum_{k} \sum_{l, l \neq k}\left(1+| | \overrightarrow{\mathbf{z}}_{k}-\overrightarrow{\mathbf{z}}_{l}| |^{2}\right)^{-1}}$。则梯度为：
$$
\nabla_{\overrightarrow{\mathbf{z}}_{i}} \mathcal{L}=\sum_{j} 4\left(p_{i, j}-q_{i, j}\right)\left(\overrightarrow{\mathbf{z}}_{i}-\overrightarrow{\mathbf{z}}_{j}\right)\left(1+\left\|\overrightarrow{\mathbf{z}}_{i}-\overrightarrow{\mathbf{z}}_{j}\right\|^{2}\right)^{-1}
$$
也可以选择自由度超过 `1` 的 `t` 分布。自由度越高，越接近高斯分布。

##### 2.3. Clustering

###### 外部指标

对于数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{2}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$，假定通过聚类给出的簇划分为$\mathcal{C}=\left\{\mathbb{C}_{1}, \mathbb{C}_{2}, \cdots, \mathbb{C}_{K}\right\}$。参考模型给出的簇划分为$\mathcal{C}^{*}=\left\{\mathbb{C}_{1}^{*}, \mathbb{C}_{2}^{*}, \cdots, \mathbb{C}_{K^{\prime}}^{*}\right\}$，其中$K$和$K^{\prime}$不一定相等 。

令$\vec{\lambda}, \vec{\lambda}^{*}$分别表示$\mathcal{C}, \mathcal{C}^{*}$的簇标记向量。定义：
$$
\begin{array}{l}{a=|S S|, S S=\left\{\left(\vec{x}_{i}, \vec{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right\}} \\ {b=|S D|, S D=\left\{\left(\vec{x}_{i}, \vec{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right\}} \\ {c=|D S|, D S=\left\{\left(\vec{x}_{i}, \vec{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right\}} \\ {d=|D D|, D D=\left\{\left(\vec{x}_{i}, \vec{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right\}}\end{array}
$$
其中$|\cdot|$表示集合的元素的个数。各集合的意义为：$SS$：包含了同时隶属于$\mathcal{C}, \mathcal{C}^{*}$的样本对。$SD$：包含了隶属于  $\mathcal{C}$，但是不隶属于$\mathcal{C}^{*}$的样本对。$DS$：包含了不隶属于 $\mathcal{C}$， 但是隶属于$\mathcal{C}^{*}$的样本对。$DD$：包含了既不隶属于 $\mathcal{C}$， 又不隶属于$\mathcal{C}^{*}$的样本对。由于每个样本对$\left(\overrightarrow{\mathbf{x}}_{i}, \overrightarrow{\mathbf{x}}_{j}\right), i<j$仅能出现在一个集合中，因此有$a+b+c+d=\frac{N(N-1)}{2}$。

$\text{Jaccard}系数$: $J C=\frac{a}{a+b+c}$

$\text{FM}指数$: $F M I=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$

$\text{Rand}指数$: $R I=\frac{a+d}{N(N-1) / 2}$

$\text{ARI}指数$:$A R I=\frac{\sum_{i} \sum_{j} C_{n_{i, j}}^{2}-\left[\sum_{i} C_{s_{i}}^{2} \times \sum_{j} C_{i_{j}}^{2}\right] / C_{N}^{2}}{\frac{1}{2}\left[\sum_{i} C_{s_{i}}^{2}+\sum_{j} C_{t_{j}}^{2}\right]-\left[\sum_{i} C_{s_{i}}^{2} \times \sum_{j} C_{t_{j}}^{2}\right] / C_{N}^{2}}$

###### 内部指标

$\text{DB}指数$：$D I=\frac{\min _{k \neq l} d_{\min }\left(\mathbb{C}_{k}, \mathbb{C}_{l}\right)}{\max _{i} \operatorname{diam}\left(\mathbb{C}_{i}\right)}$

$\text{Dunn}指数$: $D I=\frac{\min _{k \neq l} d_{\min }\left(\mathbb{C}_{k}, \mathbb{C}_{l}\right)}{\max _{i} \operatorname{diam}\left(\mathbb{C}_{i}\right)}$

##### 原型聚类

###### $\text{k-means++}$

利用Gap statistic方法啊来进行分类中心数目的确定。$Gap(K) = E(\log D_k)-\log D_k$。其中$D_k$为损失函数，$E(\log D_k)$为期望， 这个数值通常通过蒙特卡洛模拟产生，我们在样本里所在的区域中按照均匀分布随机产生和原始样本数一样多的随机样本，并对这个随机样本做 K-Means，从而得到一个$D_k$。如此往复多次，通常 20 次，我们可以得到 20 个$\log D_k$。对这 20 个数值求平均值，就得到了$E(\log D_k)$的近似值。  而 Gap statistic 取得最大值所对应的 K 就是最佳的 K。 

`k-means++` 属于 `k-means` 的变种，它主要解决`k-means` 严重依赖于分类中心初始化的问题。`k-means++` 选择初始均值向量时，尽量安排这些初始均值向量之间的距离尽可能的远。

![](D:/学习/MarkDown/picture/1/118.png)

###### $\text{k-medoids}$

![](D:/学习/MarkDown/picture/1/82.png)

###### 学习向量量化

与一般聚类算法不同，学习向量量化假设数据样本带有类别标记，学习过程需要利用样本的这些监督信息来辅助聚类。给定样本集$\mathbb{D}=\left\{\left(\overrightarrow{\mathbf{x}}_{1}, y_{1}\right),\left(\overrightarrow{\mathbf{x}}_{2}, y_{2}\right), \cdots,\left(\overrightarrow{\mathbf{x}}_{N}, y_{N}\right)\right\}, \overrightarrow{\mathbf{x}} \in \mathcal{X}, y \in \mathcal{Y}$，`LVQ`的目标是从特征空间中挑选一组样本作为原型向量$\left\{\overrightarrow{\mathbf{p}}_{1}, \overrightarrow{\mathbf{p}}_{2}, \cdots, \overrightarrow{\mathbf{p}}_{Q}\right\}$。每个原型向量代表一个聚类簇。原型向量从特征空间中取得，它们不一定就是$\mathbf{D}$中的某个样本。

`LVQ`的想法是：通过从样本中挑选一组样本作为原型向量$\left\{\overrightarrow{\mathbf{p}}_{1}, \overrightarrow{\mathbf{p}}_{2}, \cdots, \overrightarrow{\mathbf{p}}_{Q}\right\}$，可以实现对样本空间$\mathcal{X}$的簇划分。对任意样本$\overrightarrow{\mathbf{x}}$，它被划入与距离最近的原型向量所代表的簇中。对于每个原型向量$\overrightarrow{\mathbf{p}}_{q}$，它定义了一个与之相关的一个区域$\mathbf{R}_q$，该区域中每个样本与$\overrightarrow{\mathbf{p}}_{q}$的距离都不大于它与其他原型向量$\overrightarrow{\mathbf{p}}_{q^{\prime}}$的距离。
$$
\mathbf{R}_{q}=\left\{\overrightarrow{\mathbf{x}} \in \mathcal{X} |\left\|\overrightarrow{\mathbf{x}}-\overrightarrow{\mathbf{p}}_{q}\right\|_{2} \leq \min _{q \neq q}\left\|\overrightarrow{\mathbf{x}}-\overrightarrow{\mathbf{p}}_{q}\right\|_{2}\right\}
$$
区域$\left\{\overrightarrow{\mathbf{p}}_{1}, \overrightarrow{\mathbf{p}}_{2}, \cdots, \overrightarrow{\mathbf{p}}_{Q}\right\}$对样本空间$\mathcal{X}$形成了一个簇划分，该划分通常称作 `Voronoi`剖分。

![](D:/学习/MarkDown/picture/1/83.png)

##### 密度聚类

###### $\text{DBSCAN}$

`DBSCAN`是一种著名的密度聚类算法，它基于一组邻域参数$(\epsilon, \text{MinPts})$来刻画样本分布的紧密程度。给定数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{2}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$， 

定义：$\epsilon$-邻域：$N_{\epsilon}\left(\overrightarrow{\mathbf{x}}_{i}\right)=\left\{\overrightarrow{\mathbf{x}}_{j} \in \mathbb{D} | \text { distance }\left(\overrightarrow{\mathbf{x}}_{i}, \overrightarrow{\mathbf{x}}_{j}\right) \leq \epsilon\right\}$。

核心对象：若$\left|N_{\epsilon}\left(\overrightarrow{\mathbf{x}}_{i}\right)\right| \geq \text{MinPts}$，则称$\overrightarrow{\mathbf{x}}_{i}$是一个核心对象。

密度直达：若$\overrightarrow{\mathbf{x}}_{i}$是一个核心对象，且$\overrightarrow{\mathbf{x}}_{j} \in N_{\epsilon}\left(\overrightarrow{\mathbf{x}}_{i}\right)$， 则称$\overrightarrow{\mathbf{x}}_{j}$由$\overrightarrow{\mathbf{x}}_{i}$密度直达，记作$\overrightarrow{\mathbf{x}}_{i} \mapsto \overrightarrow{\mathbf{x}}_{j}$。

密度可达：对于$\overrightarrow{\mathbf{x}}_{i}$和$\overrightarrow{\mathbf{x}}_{j}$， 若存在样本序列$\left(\overrightarrow{\mathbf{p}}_{0}, \overrightarrow{\mathbf{p}}_{1}, \overrightarrow{\mathbf{p}}_{2}, \cdots, \overrightarrow{\mathbf{p}}_{m}, \overrightarrow{\mathbf{p}}_{m+1}\right)$， 其中$\overrightarrow{\mathbf{p}}_{0}=\overrightarrow{\mathbf{x}}_{i}, \overrightarrow{\mathbf{p}}_{m+1}=\overrightarrow{\mathbf{x}}_{j}, \overrightarrow{\mathbf{p}}_{s} \in \mathbb{D}$，如果$\overrightarrow{\mathbf{p}}_{s+1}$由$\overrightarrow{\mathbf{p}}_{s}$密度直达，则称$\overrightarrow{\mathbf{x}}_{j}$由$\overrightarrow{\mathbf{x}}_{i}$密度可达，记作$\overrightarrow{\mathbf{x}}_{i} \leadsto \overrightarrow{\mathbf{x}}_{j}$。

密度相连：对于$\overrightarrow{\mathbf{x}}_{i}$和$\overrightarrow{\mathbf{x}}_{j}$，若存在$\overrightarrow{\mathbf{x}}_{k}$，使得$\overrightarrow{\mathbf{x}}_{i}$与$\overrightarrow{\mathbf{x}}_{j}$均由$\overrightarrow{\mathbf{x}}_{k}$密度可达，则称$\overrightarrow{\mathbf{x}}_{i}$与$\overrightarrow{\mathbf{x}}_{i}$密度相连 ，记作$\overrightarrow{\mathbf{x}}_{i} \sim \overrightarrow{\mathbf{x}}_{j}$。

`DBSCAN`算法的簇定义：给定邻域参数$(\epsilon, \text{MinPts})$， 一个簇$\mathbb{C} \subseteq \mathbb{D}$是满足下列性质的非空样本子集：连接性： 若$\overrightarrow{\mathbf{x}}_{i} \in \mathbb{C}, \overrightarrow{\mathbf{x}}_{j} \in \mathbb{C}$，则$\overrightarrow{\mathbf{x}}_{i} \sim \overrightarrow{\mathbf{x}}_{j}$。最大性：若$\overrightarrow{\mathbf{x}}_{i} \in \mathbb{C}$，且$\overrightarrow{\mathbf{x}}_{i} \leadsto \overrightarrow{\mathbf{x}}_{j}$， 则$\overrightarrow{\mathbf{x}}_{j} \in \mathbb{C}$。

`DBSCAN`算法的思想：若$\vec{\mathbf{x}}$为核心对象，则  密度可达的所有样本组成的集合记作$\mathbb{X}=\left\{\overrightarrow{\mathbf{x}}^{\prime} \in \mathbb{D} | \overrightarrow{\mathbf{x}} \leadsto \overrightarrow{\mathbf{x}}^{\prime}\right\}$。可以证明 ：$\mathbb{X}$就是满足连接性与最大性的簇。于是 `DBSCAN`算法首先任选数据集中的一个核心对象作为种子`seed`，再由此出发确定相应的聚类簇。

###### $\text{Mean-Shift}$算法

![](D:/学习/MarkDown/picture/1/84.png)

##### 层次聚类

层次聚类试图在不同层次上对数据集进行划分，从而形成树形的聚类结构。

###### $\text{BIRCH}$算法

聚类特征`CF`：每个`CF` 都是刻画一个簇的特征的三元组：$C F=\left(\mathrm{num}, \vec{\Sigma}_{l}, \Sigma_{s}\right)$。其中：$\text{num}$：表示簇内样本数量的数量。$\vec{\Sigma}_{l}$：表示簇内样本的线性求和：$\vec{\Sigma}_{l}=\sum_{\vec{x}_{i} \in \mathbb{S}} \overrightarrow{\mathbf{x}}_{i}$。$\Sigma_{s}$：表示簇内样本的长度的平方和。$\Sigma_{s}=\sum_{\vec{x}_{i} \in \mathbb{S}}\left\|\vec{x}_{i}\right\|^{2}=\sum_{\vec{x}_{i} \in \mathbb{S}} \overrightarrow{\mathbf{x}}_{i}^{T} \overrightarrow{\mathbf{x}}_{i}$。根据`CF` 的定义可知：如果`CF1` 和 `CF2` 分别表示两个不相交的簇的特征，如果将这两个簇合并成一个大簇，则大簇的特征为：$C F_{m e r g e}=C F_{1}+C F_{2}$。

`CF`树的结构类似于平衡`B+`树 。树由三种结点构成：根结点、中间结点、叶结点。根结点、中间结点：由若干个聚类特征`CF` ，以及这些`CF` 指向子结点的指针组成。叶结点：由若干个聚类特征`CF` 组成。叶结点没有子结点，因此`CF` 没有指向子结点的指针。所有的叶结点通过双向链表连接起来。在`BIRCH` 算法结束时，叶结点的每个`CF` 对应的样本集就对应了一个簇。`CF` 树有三个关键参数：枝平衡因子$\beta$：非叶结点中，最多不能包含超过$\beta$个 `CF` 。叶平衡因子$\lambda$：叶结点中，最多不能包含超过$\lambda$个 `CF` 。空间阈值$\tau$：叶结点中，每个`CF` 对应的子簇的大小不能超过$\tau$。由于`CF` 的可加性，所以`CF` 树中，每个父结点的`CF` 等于它所有子结点的所有`CF` 之和。

##### 谱聚类

谱聚类的主要思想是：基于数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$来构建图$\mathcal{G}=(\mathbb{V}, \mathbb{E})$，其中：顶点$\mathbb{V}$：由数据集中的数据点组成：$\mathbb{V}=\{1,2, \cdots, N\}$。边$\mathbb{E}$：任意一对顶点之间存在边。距离越近的一对顶点，边的权重越高；距离越远的一对顶点，边的权重越低。通过对图$\mathcal{G}$进行切割，使得切割之后：不同子图之间的边的权重尽可能的低、各子图内的边的权重尽可能的高。这样就完成了聚类。

在图$\mathcal{G}=(\mathbb{W}, \mathbb{E})$中，定义权重$\omega_{i,j}$为顶点$i$和$j$之间的权重，其中$i, j \in \mathbb{V}$。定义$\mathbf{W}=\left(w_{i, j}\right)_{N \times N}$为邻接矩阵由于$\mathcal{G}$为无向图，因此$w_{i, j}=w_{j, i}$。即：$\mathbf{W}=\mathbf{W}^{T}$。对图中顶点$i$，定义它的度$d_i$为：所有与顶点$i$相连的边的权重之和：$d_{i}=\sum_{j=1}^{N} w_{i, j}$。定义度矩阵$\mathbf{D}$为一个对角矩阵，其中对角线分别为各顶点的度。对于顶点集合$\mathbb{V}$的一个子集$\mathbb{A}\subset \mathbb{V}$，定义$|\mathbb{A}|$为子集$\mathbb{A}$中点的个数；定义$\operatorname{vol}(\mathbb{A})=\sum_{i \in \mathbb{A}} d_{i}$，为子集$\mathbb{A}$中所有点的度之和。事实上在谱聚类中，通常只给定数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$，因此需要计算出邻接矩阵$\mathbf{W}$。基本思想是：距离较近的一对点，边的权重较高；距离较远的一对点，边的权重较低。基本方法是：首先构建相似度矩阵$\mathbf{S}=\left(s_{i, j}\right)_{N \times N}$，然后使用$\epsilon$-近邻法、$K$近邻法、或者全连接法。

$\epsilon$-近邻法：设置一个距离阈值$\epsilon$，定义邻接矩阵$\mathbf{W}$为：
$$
w_{i, j}=\left\{\begin{array}{ll}{0,} & {s_{i, j}>\varepsilon} \\ {\varepsilon,} & {s_{i, j} \leq \varepsilon}\end{array}\right.
$$
$K$近邻法：利用 `KNN` 算法选择每个样本最近的$K$个点作为近邻，其它点与当前点之间的边的权重为 0 。这种做法会导致邻接矩阵$\mathbf{W}$非对称，因为当$\overrightarrow{\mathbf{x}}_{j}$是$\overrightarrow{\mathbf{x}}_{i}$的$K$近邻时， $\overrightarrow{\mathbf{x}}_{i}$不一定是$\overrightarrow{\mathbf{x}}_{j}$的$K$近邻。为了解决对称性问题，有两种做法：只要一个点在另一个点的  近邻中，则认为是近邻。即：取并集。
$$
w_{i, j}=w_{j, i}=\left\{\begin{array}{ll}{0,} & {\overrightarrow{\mathbf{x}}_{i} \notin K N N\left(\overrightarrow{\mathbf{x}}_{j}\right) \text { and } \overrightarrow{\mathbf{x}}_{j} \notin K N N\left(\overrightarrow{\mathbf{x}}_{i}\right)} \\ {s_{i, j},} & {\overrightarrow{\mathbf{x}}_{i} \in K N N\left(\overrightarrow{\mathbf{x}}_{j}\right) \text { or } \overrightarrow{\mathbf{x}}_{j} \in K N N\left(\overrightarrow{\mathbf{x}}_{i}\right)}\end{array}\right.
$$
只有两个点互为对方的  近邻中，则认为是近邻。即：取交集。
$$
w_{i, j}=w_{j, i}=\left\{\begin{array}{ll}{0,} & {\overrightarrow{\mathbf{x}}_{i} \notin K N N\left(\overrightarrow{\mathbf{x}}_{j}\right) \text { or } \overrightarrow{\mathbf{x}}_{j} \notin K N N\left(\overrightarrow{\mathbf{x}}_{i}\right)} \\ {s_{i, j},} & {\overrightarrow{\mathbf{x}}_{i} \in K N N\left(\overrightarrow{\mathbf{x}}_{j}\right) \text { and } \overrightarrow{\mathbf{x}}_{j} \in K N N\left(\overrightarrow{\mathbf{x}}_{i}\right)}\end{array}\right.
$$
全连接法：所有点之间的权重都大于 0 ：$w_{i, j}=s_{i, j}$。

定义拉普拉斯矩阵$\mathbf{L}=\mathbf{D}-\mathbf{W}$，其中$\mathbf{D}$为度矩阵、$\mathbf{W}$为邻接矩阵。拉普拉斯矩阵$\mathbf{L}$的性质：$\mathbf{L}$是对称矩阵。因为$\mathbf{L}$是实对称矩阵，因此它的特征值都是实数。对任意向量$\overrightarrow{\mathbf{f}}=\left(f_{1}, f_{2}, \cdots, f_{N}\right)^{T}$，有：$\overrightarrow{\mathbf{f}}^{T} \mathbf{L} \overrightarrow{\mathbf{f}}=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} w_{i, j}\left(f_{i}-f_{j}\right)^{2}$。$\mathbf{L}$是半正定的，且对应的$N$个特征值都大于等于0，且最小的特征值为 0。

给定无向图$\mathcal{G}=(\mathbb{V}, \mathbb{E})$，设子图的点的集合$\mathbb{A}$和子图的点的集合$\mathbb{B}$都是$\mathbb{V}$的子集，且$\mathbb{A} \cap \mathbb{B}=\phi$。定义$\mathbb{A}$和$\mathbb{B}$之间的切图权重为：$W(\mathbb{A}, \mathbb{B})=\sum_{i \in \mathbb{A}, j \in \mathbb{B}} w_{i, j}$。对于无向图$\mathcal{G}=(\mathbb{V}, \mathbb{E})$，假设将它切分为$k$个子图：每个子图的点的集合为$\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}$，满足$\mathbb{A}_{i} \cap \mathbb{A}_{j}=\phi, i \neq j$且$\mathbb{A}_{1} \cup \cdots \cup \mathbb{A}_{k}=\mathbb{V}$。定义切图`cut` 为：$\operatorname{cut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)=\sum_{i=1}^{k} W\left(\mathbb{A}_{i}, \overline{\mathbb{A}}_{i}\right)$，其中$\overline{\mathbb{A}}$为$\mathbb{A}$的补集。

###### 最小切图

引入指示向量$\overrightarrow{\mathbf{q}}_{j}=\left(q_{j, 1}, \cdots, q_{j, N}\right)^{T}, j=1,2, \cdots, k$，定义：$q_{j, i}=\left\{\begin{array}{ll}{0,} & {i \notin \mathbb{A}_{j}} \\ {1,} & {i \in \mathbb{A}_{j}}\end{array}\right.$

因此$\operatorname{cut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)=\sum_{j=1}^{k} \overrightarrow{\mathbf{q}}_{j}^{T} \mathbf{L} \overrightarrow{\mathbf{q}}_{j}=\operatorname{tr}\left(\mathbf{Q}^{T} \mathbf{L} \mathbf{Q}\right)$。其中$\mathbf{Q}=\left(\overrightarrow{\mathbf{q}}_{1}, \cdots, \overrightarrow{\mathbf{q}}_{k}\right)$，$tr(\cdot)$为矩阵的迹。考虑到顶点$i$有且仅位于一个子图中，则有约束条件：
$$
q_{j, m} \in\{0,1\}, \quad \overrightarrow{\mathbf{q}}_{i} \cdot \overrightarrow{\mathbf{q}}_{j}=\left\{\begin{array}{ll}{0,} & {i \neq j} \\ {|\hat{A}|_{j},} & {i=j}\end{array}\right.
$$
最小切图算法：$\operatorname{cut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)$最小的切分。即求解：
$$
\begin{array}{c}{\min _{\mathbf{Q}} \operatorname{tr}\left(\mathbf{Q}^{T} \mathbf{L} \mathbf{Q}\right)} \\ {\text {s.t. } q_{j, m} \in\{0,1\}, \quad \overrightarrow{\mathbf{q}}_{i} \cdot \overrightarrow{\mathbf{q}}_{j}=\left\{\begin{array}{ll}{0,} & {i \neq j} \\ {|\hat{A}|_{j},} & {i=j}\end{array}\right.}\end{array}
$$

###### $\text{RatioCut}$算法

`RatioCut` 切图不仅考虑最小化$\operatorname{cut}\left(A_{1}, \cdots, \mathbb{A}_{k}\right)$，它还考虑最大化每个子图的点的个数。即：$\operatorname{Ratio} \operatorname{Cut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)=\sum_{i=1}^{k} \frac{W\left(\mathbb{A}_{i}, \overline{A}_{i}\right)}{\left|\mathbb{A}_{i}\right|}$。

引入指示向量$\overrightarrow{\mathbf{h}}_{j}=\left(h_{j, 1}, \cdots, h_{j, N}\right)^{T}, j=1,2, \cdots, k$，定义$h_{j, i}=\left\{\begin{array}{ll}{0,} & {i \notin \mathbb{A}_{j}} \\ {\frac{1}{\sqrt{\left|\hat{A}_{j}\right|}},} & {i \in \mathbb{A}_{j}}\end{array}\right.$。因此$\text{RatioCut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)=\sum_{j=1}^{k} \overrightarrow{\mathbf{h}}_{j}^{T} \mathbf{L} \overrightarrow{\mathbf{h}}_{j}=\operatorname{tr}\left(\mathbf{H}^{T} \mathbf{L H}\right)$。其中$\mathbf{H}=\left(\overrightarrow{\mathbf{h}}_{1}, \cdots, \overrightarrow{\mathbf{h}}_{k}\right), t r(\cdot)$为矩阵的迹。考虑到顶点$I$有且仅位于一个子图中，则有约束条件：
$$
\overrightarrow{\mathbf{h}}_{i} \cdot \overrightarrow{\mathbf{h}}_{j}=\left\{\begin{array}{ll}{0,} & {i \neq j} \\ {1,} & {i=j}\end{array}\right.
$$
`RatioCut`算法：$\text{RatioCut} \left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)$最小的切分。即求解：
$$
\begin{array}{l}{\min _{\mathbf{H}} \operatorname{tr}\left(\mathbf{H}^{T} \mathbf{L H}\right)} \\ {\text {s.t.}\mathbf{H}^{T} \mathbf{H}=\mathbf{I}}\end{array}
$$

###### $\text{Ncut}$算法

`Ncut` 切图不仅考虑最小化 ，它还考虑最大化每个子图的边的权重。即：
$$
\text{Ncut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)=\sum_{i=1}^{k} \frac{W\left(\mathbb{A}_{i}, \overline{A}_{i}\right)}{\operatorname{vol}\left(\mathbb{A}_{i}\right)}
$$
引入指示向量$\overrightarrow{\mathbf{h}}_{j}=\left(h_{j, 1}, \cdots, h_{j, N}\right)^{T}, j=1,2, \cdots, k$，定义：$h_{j, i}=\left\{\begin{array}{ll}{0,} & {i \notin \mathbb{A}_{j}} \\ {\frac{1}{\sqrt{v o l\left(k_{j}\right)}},} & {i \in \mathbb{A}_{j}}\end{array}\right.$。考虑到顶点$i$有且仅位于一个子图中，则有约束条件：
$$
\overrightarrow{\mathbf{h}}_{i} \cdot \overrightarrow{\mathbf{h}}_{j}=\left\{\begin{array}{ll}{0,} & {i \neq j} \\ {\frac{1}{v o l\left(A_{j}\right)},} & {i=j}\end{array}\right.
$$
`Ncut`算法：$\text{Ncut}\left(\mathbb{A}_{1}, \cdots, \mathbb{A}_{k}\right)$最小的切分。即求解
$$
\begin{array}{c}{\min _{\mathbf{H}} \operatorname{tr}\left(\mathbf{H}^{T} \mathbf{L} \mathbf{H}\right)} \\ {\text { s.t. } \mathbf{H}^{T} \mathbf{D} \mathbf{H}=\mathbf{I}}\end{array}
$$

##### 2.5. Decomposing signal in components

在高维情形下出现的数据样本稀疏、距离计算困难等问题是所有机器学习方法共同面临的严重障碍，称作维度灾难。降维之所以有效的原因是：人们观测或者收集到的数据样本虽然是高维的，但是与学习任务密切相关的也许仅仅是某个低维分布，即高维空间中的一个低维嵌入。

###### $\text{PCA}$

`PCA`降维的准则有两个：最近重构性：样本集中所有点，重构后的点距离原来的点的误差之和最小。最大可分性：样本点在低维空间的投影尽可能分开。

PCA centers but does not scale the input data for each feature before applying the SVD. The optional parameter `whiten=True` makes it possible to project the data onto the singular space while scaling each component to unit variance. This is often useful if the models down-stream make strong assumptions on the isotropy of the signal.

###### $\text{SVD}$

酉矩阵：若$n$阶矩阵满足$\mathbf{U}^H\mathbf{U}=\mathbf{U}\mathbf{U}^H=\mathbf{I}$，则它是酉矩阵。其中$\mathbf{U}^H$为$\mathbf{U}$的共轭转置。$\mathbf{U}$为酉矩阵的充要条件是：$\mathbf{U}^H=\mathbf{U}^{-1}$。

奇异值分解：设$\mathbf{X}为$阶$N\times n$矩阵，且$rank(X)=r$，则存在$N$阶酉矩阵$\mathbf{V}$和$n$阶酉矩阵$\mathbf{U}$，使得：$\mathbf{V}^{H} \mathbf{X} \mathbf{U}=\left[ \begin{array}{ll}{\Sigma} & {\mathbf{0}} \\ {\mathbf{0}} & {\mathbf{0}}\end{array}\right]_{N \times n}$，其中
$$
\Sigma=\left[ \begin{array}{ccccc}{\sigma_{1}} & {0} & {0} & {\cdots} & {0} \\ {0} & {\sigma_{2}} & {0} & {\cdots} & {0} \\ {\vdots} & {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {0} & {0} & {0} & {\cdots} & {\sigma_{r}}\end{array}\right]
$$
根据酉矩阵的性质, $\mathbf{V} \mathbf{V}^{H}=\mathbf{I}_{N \times N}, \mathbf{U} \mathbf{U}^{H}=\mathbf{I}_{n \times n}$，则有：$\mathbf{X}=\mathbf{V} \left[ \begin{array}{ll}{\Sigma} & {\mathbf{0}} \\ {\mathbf{0}} & {\mathbf{0}}\end{array}\right]_{N \times n} \mathbf{U}^{H} \Longrightarrow \mathbf{X}^{H}=\mathbf{U} \left[ \begin{array}{ll}{\Sigma} & {\mathbf{0}} \\ {\mathbf{0}} & {\mathbf{0}}\end{array}\right]_{n \times N} \mathbf{V}^{H}$

则有$\mathbf{X}^{H} \mathbf{X}=\mathbf{U M U}^{H}$， 其中$\mathbf{M}$是个$n$阶对角矩阵：由数据集$\mathbb{D}$中样本构成的$\mathbf{X}$为实矩阵，因此有$\mathbf{X}^H=\mathbf{X}^T$。另外考虑到$\mathbf{X}^T\mathbf{X}$为实对称矩阵，因此$\mathbf{V}$也是实矩阵，因此$\mathbf{U}^H=\mathbf{U}^T$。 则有：$\mathbf{X}^{T} \mathbf{X}=\mathbf{U} \mathbf{M} \mathbf{U}^{T}$。根据$\mathbf{U} \mathbf{U}^{T}=\mathbf{I}$，则有：$\mathbf{X}^{T} \mathbf{X U}=\mathbf{U} \mathbf{M}$。根据$\mathbf{M}$是个对角矩阵的性质，有：$\mathbf{U M}=\mathbf{M U}$，则有：$\mathbf{X}^{T} \mathbf{X} \mathbf{U}=\mathbf{M} \mathbf{U}$。则$\lambda_{i}, i=1,2, \cdots, r$就是的$\mathbf{X}^{T} \mathbf{X}$特征值， 其对应的单位特征向量组成正交矩阵$\mathbf{U}$。因此`SVD`奇异值分解等价于`PCA`主成分分析，核心都是求解$\mathbf{X}^{T} \mathbf{X}$的特征值以及对应的单位特征向量。

###### $\text{KPCA}$

假定原始特征空间中的样本点$\vec{\mathbf{x}}_{i}$通过映射$\phi$映射到高维特征空间的坐标为$\vec{\mathbf{x}}_{i,\phi}$，即$\vec{\mathbf{x}}_{i, \phi}=\phi\left(\vec{\mathbf{x}}_{i}\right)$。且假设高维特征空间为$n$维的，即：$\vec{\mathbf{x}}_{i, \phi} \in \mathbb{R}^{n}$。假定要将高维特征空间中的数据投影到低维空间中，投影矩阵为$\mathbf{W}$为$n\times d$维矩阵。根据 `PCA` 推导的结果，求解方程：$\mathbf{X}_{\phi}^{T} \mathbf{X}_{\phi} \mathbf{W}=\lambda \mathbf{W}$。其中$\mathbf{X}_{\phi}=\left(\vec{\mathbf{x}}_{1, \phi}^{T}, \vec{\mathbf{x}}_{2, \phi}^{T}, \cdots, \vec{\mathbf{x}}_{N, \phi}^{T}\right)^{T}$为$N\times n$维矩阵。于是有：$\left(\sum_{i=1}^{N} \phi\left(\vec{\mathbf{x}}_{i}\right) \phi\left(\vec{\mathbf{x}}_{i}\right)^{T}\right) \mathbf{W}=\lambda \mathbf{W}$。

定义核矩阵 ：
$$
\mathbf{K}=\left[ \begin{array}{cccc}{\kappa\left(\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{1}\right)} & {\kappa\left(\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{2}\right)} & {\cdots} & {\kappa\left(\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{N}\right)} \\ {\kappa\left(\overrightarrow{\mathbf{x}}_{2}, \overrightarrow{\mathbf{x}}_{1}\right)} & {\kappa\left(\overrightarrow{\mathbf{x}}_{2}, \overrightarrow{\mathbf{x}}_{2}\right)} & {\cdots} & {\kappa\left(\overrightarrow{\mathbf{x}}_{2}, \overrightarrow{\mathbf{x}}_{N}\right)} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\overrightarrow{\mathbf{x}}_{N}, \overrightarrow{\mathbf{x}}_{1}\right)} & {\kappa\left(\overrightarrow{\mathbf{x}}_{N}, \overrightarrow{\mathbf{x}}_{2}\right)} & {\cdots} & {\kappa\left(\overrightarrow{\mathbf{x}}_{N}, \overrightarrow{\mathbf{x}}_{N}\right)}\end{array}\right]
$$
则有：$\mathbf{X}_{\phi} \mathbf{X}_{\phi}^{T}=\mathbf{K}$ 。定义$\vec{\alpha}_{i}=\frac{\vec{\mathbf{x}}_{i, \phi}^{T} \mathbf{w}}{\lambda}$，则$\vec{\alpha}_{i}$为$1\times d$维行向量 。定义：$\mathbf{A}=\left(\vec{\alpha}_{1}, \vec{\alpha}_{2}, \cdots, \vec{\alpha}_{N}\right)^{T}$，则有：
$$
\mathbf{W}=\frac{1}{\lambda}\left(\sum_{i=1}^{N} \overrightarrow{\mathbf{x}}_{i, \phi} \overrightarrow{\mathbf{x}}_{i, \phi}^{T}\right) \mathbf{W}=\sum_{i=1}^{N} \overrightarrow{\mathbf{x}}_{i, \phi} \frac{\overrightarrow{\mathbf{x}}_{i, \phi}^{T} \mathbf{W}}{\lambda}=\sum_{i=1}^{N} \overrightarrow{\mathbf{x}}_{i, \phi} \vec{\alpha}_{i}=\mathbf{X}_{\phi}^{T} \mathbf{A}
$$
将$\mathbf{W}=\mathbf{X}_{\phi}^{T} \mathbf{A}$代入$\mathbf{x}_{\phi}^{T} \mathbf{X}_{\phi} \mathbf{W}=\lambda \mathbf{W}$，有：$\mathbf{X}_{\phi}^{T} \mathbf{X}_{\phi} \mathbf{X}_{\phi}^{T} \mathbf{A}=\lambda \mathbf{X}_{\phi}^{T} \mathbf{A}$。同样该问题也是一个特征值分解问题，取$\mathbf{K}$最大的$d$个特征值对应的特征向量组成  即可。对于新样本$\vec{\mathbf{x}}$， 其投影后第$j$维的坐标为：
$$
z_{j}=\overrightarrow{\mathbf{w}}_{j}^{T} \phi(\overrightarrow{\mathbf{x}})=\sum_{i=1}^{N} \alpha_{i, j} \phi\left(\overrightarrow{\mathbf{x}}_{i}\right)^{T} \phi(\overrightarrow{\mathbf{x}})=\sum_{i=1}^{N} \alpha_{i, j} \kappa\left(\overrightarrow{\mathbf{x}}_{i}, \overrightarrow{\mathbf{x}}\right)
$$

###### 概率`PCA`

定义隐变量$\overrightarrow{\mathbf{z}} \in \mathbb{R}^{d}$，它属于低维空间。假设$\overrightarrow{\mathbf{z}}$的先验分布为高斯分布：$p(\overrightarrow{\mathbf{z}})=\mathcal{N}(\overrightarrow{\mathbf{0}}, \mathbf{I})$。定义观测变量$\overrightarrow{\mathbf{x}} \in \mathbb{R}^{n}$，它属于高维空间。假设条件概率分布$p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}})$也是高斯分布：$p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}})=\mathcal{N}\left(\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}, \sigma^{2} \mathbf{I}\right)$。则`PPCA` 模型生成观测样本的步骤为：首先以概率$p(\overrightarrow{\mathbf{z}})$生成隐变量$\overrightarrow{\mathbf{z}}$。然后观测样本$\vec{\mathbf{x}}$由如下规则生成：$\overrightarrow{\mathbf{x}}=\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}+\vec{\epsilon}$。

可以利用最大似然准则来确定参数$\mathbf{W}, \vec{\mu}, \sigma^{2}$的解析解。根据边缘概率分布的定义有：$p(\overrightarrow{\mathbf{x}})=\int p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}}) d \overrightarrow{\mathbf{z}}$。由于$p(\overrightarrow{\mathbf{z}}), p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}})$均为高斯分布，因此$p(\overrightarrow{\mathbf{x}})$也是高斯分布。假$\overrightarrow{\mathbf{x}} $的其均值为$\mu^{\prime}$，协方差为$\mathbf{C}$。则：
$$
\begin{array}{c}{\vec{\mu}^{\prime}=\mathbb{E}[\overrightarrow{\mathbf{x}}]=\mathbb{E}[\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}+\vec{\epsilon}]=\vec{\mu}} \\ {\mathbf{C}=\operatorname{cov}[\overrightarrow{\mathbf{x}}]=\mathbb{E}\left[(\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}+\vec{\epsilon})(\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}+\vec{\epsilon})^{T}\right]} \\ {=\mathbb{E}\left[\mathbf{W} \overrightarrow{\mathbf{z}}^{T} \mathbf{W}\right]+\mathbb{E}\left[\vec{\epsilon} \vec{\epsilon}^{T}\right]+\vec{\mu} \vec{\mu}^{T}=\mathbf{W} \mathbf{W}^{T}+\sigma^{2} \mathbf{I}+\vec{\mu} \vec{\mu}^{T}}\end{array}
$$
因此$p(\overrightarrow{\mathbf{x}})=\mathcal{N}(\vec{\mu}, \mathbf{C})$。

给定数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$，则对数似然函数为：
$$
\begin{array}{c}{\mathcal{L}=\log p\left(\mathbb{D} ; \mathbf{W}, \vec{\mu}, \sigma^{2}\right)=\sum_{i=1}^{N} \log p\left(\overrightarrow{\mathbf{x}}_{i} ; \mathbf{W}, \vec{\mu}, \sigma^{2}\right)} \\ {=-\frac{N n}{2} \log (2 \pi)-\frac{N}{2} \log |\mathbf{C}|-\frac{1}{2} \sum_{i=1}^{N}\left(\overrightarrow{\mathbf{x}}_{i}-\vec{\mu}\right)^{T} \mathbf{C}^{-1}\left(\overrightarrow{\mathbf{x}}_{i}-\vec{\mu}\right)}\end{array}
$$
其中$|\cdot|$这里表示行列式的值。对数据集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$进行零均值化，即：$$
\overrightarrow{\mathbf{x}}_{i} \leftarrow \overrightarrow{\mathbf{x}}_{i}-\vec{\mu}=\overrightarrow{\mathbf{x}}_{i}-\overline{\overrightarrow{\mathbf{x}}}
$$。则有：$\overrightarrow{\mathbf{x}}=\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\epsilon}$，因此$p(\overrightarrow{\mathbf{x}})=\mathcal{N}(\overrightarrow{\mathbf{x}} ; \overrightarrow{\mathbf{0}}, \mathbf{C})$。

对数似然函数：$\mathcal{L}=\log p\left(\mathbb{D} ; \mathbf{W}, \vec{\mu}, \sigma^{2}\right)=-\frac{N}{2} \log |\mathbf{C}|-\frac{1}{2} \sum_{i=1}^{N} \overrightarrow{\mathbf{x}}_{i}^{T} \mathbf{C}^{-1} \overrightarrow{\mathbf{x}}_{i}=-\frac{N}{2}\left[\log |\mathbf{C}|+\operatorname{tr}\left(\mathbf{C}^{-1} \mathbf{S}\right)\right]$

其中$\mathbf{S}=\mathbf{X}^{T} \mathbf{X}$为协方差矩阵。

$\mathcal{L}$的所有驻点都可以写做：$\mathbf{W}=\mathbf{U}_{d}\left(\mathbf{\Lambda}_{d}-\sigma^{2} \mathbf{I}\right)^{1 / 2} \mathbf{R}$。其中：$\mathbf{U}_{d} \in \mathbb{R}^{n \times d}$的列由协方差矩阵$\mathbf{S}$的任意$d$个特征向量组成。$\boldsymbol{\Lambda}_{d} \in \mathbb{R}^{d \times d}$是对角矩阵，其元素是协方差矩阵$\mathbf{S}$对应的$d$个特征值$\lambda_i$。$\mathbf{R} \in \mathbb{R}^{d \times d}$是任意一个正交矩阵。当$d$个特征向量被选择为前$d$个最大的特征值对应的特征向量时，$\mathcal{L}$取得最大值。其它的所有解都是鞍点。

假定协方差矩阵$\mathbf{S}$的特征值从大到小排列$\lambda_{1} \geq \lambda_{2} \geq \cdots \geq \lambda_{n}$，对应的$n$个特征向量为$\overrightarrow{\mathbf{u}}_{1}, \cdots, \overrightarrow{\mathbf{u}}_{n}$。则最大似然准则得到的解析解为：$\mathbf{U}=\left(\overrightarrow{\mathbf{u}}_{1}, \cdots, \overrightarrow{\mathbf{u}}_{d}\right)$，它由前$d$个特征向量组成。$\mathbf{W}=\mathbf{U}_{d}\left(\mathbf{\Lambda}_{d}-\sigma^{2} \mathbf{I}\right)^{1 / 2} \mathbf{R}$。$\sigma^{2}=\frac{1}{n-d} \sum_{i=d+1}^{n} \lambda_{i}$，它就是与丢弃的维度相关连的平均方差。$\mathbf{R}$是正交矩阵，因此它可以视作$d$维隐空间的一个旋转矩阵。

根据$\mathbf{C}=\mathbf{W} \mathbf{W}^{T}+\sigma^{2} \mathbf{I}=\mathbf{U}_{d}\left(\mathbf{\Lambda}_{d}-\sigma^{2} \mathbf{I}\right) \mathbf{U}_{d}^{T}+\sigma^{2} \mathbf{I}$，则$\mathbf{C}$与$\mathbf{R}$无关。这表明：$p(\overrightarrow{\mathbf{x}})$在隐空间中具有旋转不变性，因此$\mathbf{R}$可以选任意一个正交矩阵。

###### 因子分析

因子分析是一个线性高斯隐变量模型，它与 `PPCA` 密切相关。因子分析的定义与`PPCA` 唯一差别是：给定隐变量$\vec{\mathbf{z}}$的条件下，观测变量$\vec{\mathbf{x}}$的条件概率分布的协方差矩阵是一个对角矩阵，而不是一个各向同性的协方差矩阵。即：$p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}})=\mathcal{N}(\mathbf{W} \overrightarrow{\mathbf{z}}+\vec{\mu}, \mathbf{\Psi})$，其中$\Psi$是一个$n \times n$的对角矩阵。因此也可以认为`PPCA` 是一种特殊情形的因子分析。如果对$\vec{\mathbf{x}}$进行了零均值化，则$p(\overrightarrow{\mathbf{x}} | \overrightarrow{\mathbf{z}})=\mathcal{N}(\mathbf{W} \overrightarrow{\mathbf{z}}, \mathbf{\Psi})$。与 `PPCA` 模型相同，因子分析模型假设在给定隐变量$\vec{\mathbf{z}}$的条件下，观测变量$\vec{\mathbf{x}}$的各分量$x_{1}, x_{2}, \cdots, x_{n}$是独立的。

可以使用最大似然法来确定因子分析模型中的参数$\mathbf{W}, \mathbf{\Psi}$的值。此时$\mathbf{W}, \mathbf{\Psi}$的最大似然解不再具有解析解，因此必须用梯度下降法或者`EM` 算法迭代求解。`E` 步：用旧的参数求期望：
$$
\begin{array}{c}
\mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right]=\mathbf{G} \mathbf{W}^{T} \mathbf{\Psi}^{-1} \overrightarrow{\mathbf{x}}_{i}\\
\mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i} \overrightarrow{\mathbf{z}}_{i}^{T}\right]=\operatorname{cov}\left[\overrightarrow{\mathbf{z}}_{i}\right]+\mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right] \mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right]^{T}=\mathbf{G}+\mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right] \mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right]^{T}
\end{array}
$$
其中$\mathbf{G}=\left(\mathbf{I}+\mathbf{W}^{T} \mathbf{\Psi}^{-1} \mathbf{W}\right)^{-1}$。这里使用一个$d \times d$的矩阵求逆表达式，而不是$n \times n$的表达式。`M` 步：求最大化来获取新的参数。
$$
\begin{array}{l}{\mathbf{W}_{n e w} \leftarrow\left[\sum_{i=1}^{N} \overrightarrow{\mathbf{x}}_{i} \mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right]^{T}\right]\left[\sum_{i=1}^{N} \mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i} \overrightarrow{\mathbf{z}}_{i}^{T}\right]\right]^{-1}} \\ {\mathbf{\Psi}_{n e w} \leftarrow \operatorname{diag}\left[\mathbf{S}-\mathbf{W}_{n e w} \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}\left[\overrightarrow{\mathbf{z}}_{i}\right] \overrightarrow{\mathbf{x}}_{i}^{T}\right]}\end{array}
$$
其中$\text{diag}$将所有非对角线上的元素全部设置为零。

###### 独立成分分析

独立成分分析`ICA` 用于从混合信号中分离出原始信号。本质上它并不是一个降维的算法，而是一个信号分离算法。

假设酒会上有$n$个人，他们可以同时说话。房间里散落了$n$个声音接收器用于记录声音。酒会过后，从$n$个声音接收器中采集到一组数据：
$$
\begin{array}{c}{\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \overrightarrow{\mathbf{x}}_{2}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}} \\ {\overrightarrow{\mathbf{x}}_{i}=\left(x_{i, 1}, x_{i, 2}, \cdots, x_{i, n}\right)^{T}}\end{array}
$$
任务的目标是：从这$N$个时刻的采样数据中恢复出每个人说话的信号。这个过程也称作盲信号分离。随机变量$\vec{\mathbf{x}}$表示观测随机变量，$\vec{\mathbf{x}}_i$是其第$i$个采样值，其物理意义为：在时刻$i$采集到的$n$个声音信号。

定义：第$i$个人说话的信号为$s_i$。它是一个随机变量，其分布为$p_s(s_i)$。 为$s_{1,i}, \cdots, s_{N, i}$的$N$个时刻的采样，记作$\overrightarrow{\mathbf{u}}_{i}^{(s)}$。$n$个人说话的信号为$\overrightarrow{\mathbf{s}}=\left(s_{1}, s_{2}, \cdots, s_{n}\right)^{T}$。它是一个  维随机变量，分布为$p_s(\vec{\mathbf{s}})$。$\overrightarrow{\mathbf{s}}_{1}, \cdots, \overrightarrow{\mathbf{s}}_{N}$为 的$N$个时刻的采样。第$i$个声音接收器收到的信号为$x_i$。它是一个随机变量，其分布为$p_x(x_i)$。 $x_{1,i}, \cdots, x_{N, i}$为$x_i$的$N$个时刻的采样，记作$\overrightarrow{\mathbf{u}}_{i}^{(x)}$。$n$个声音接收器收到的信号为$\overrightarrow{\mathbf{x}}=\left(x_{1}, x_{2}, \cdots, x_{n}\right)^{T}$。它是一个$n$维随机变量，分布为$p_{x}(\overrightarrow{\mathbf{x}}) $。$\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}$。 为$\vec{\mathbf{x}}$的$N$个时刻的采样。定义矩阵$\mathbf{X}$和矩阵$\mathbf{S}$为：
$$
\mathbf{X}=\left[\begin{array}{c}{\overrightarrow{\mathbf{x}}_{1}^{T}} \\ {\vdots} \\ {\overrightarrow{\mathbf{x}}_{N}^{T}}\end{array}\right]
\mathbf{S}=\left[\begin{array}{c}{\overrightarrow{\mathbf{s}}_{1}^{T}} \\ {\vdots} \\ {\overrightarrow{\mathbf{s}}_{N}^{T}}\end{array}\right]
$$
$\mathbf{A}=\left(a_{i, j}\right)_{n \times n}$是一个未知的混合矩阵，它用于叠加$n$个人说话的信号。则有：$\mathbf{X}=\mathbf{S} \mathbf{A}^{T}$。即：$\overrightarrow{\mathbf{x}}=\mathbf{A} \overrightarrow{\mathbf{s}}$。

现在$\mathbf{X}$是已知的，即信号$\vec{\mathbf{X}}$是已知的。令$\mathbf{W}=\mathbf{A}^{-1}$，则有：$\overrightarrow{\mathbf{s}}=\mathbf{W} \overrightarrow{\mathbf{x}}$。$\mathbf{W}$称作分离矩阵。如果没有任何先验知识，则无法求解。这里需要假设$p_s(s_i)$。首先，不能选取高斯分布。其次，考虑到概率密度函数由累计分布函数求导得到，一个方便的选择是：选择累计分布函数为 `sigmoid` 函数。

给定采样样本集$\mathbb{D}=\left\{\overrightarrow{\mathbf{x}}_{1}, \cdots, \overrightarrow{\mathbf{x}}_{N}\right\}$，则对数似然函数为：
$$
\mathcal{L}=\sum_{i=1}^{N} \log p_{x}\left(\overrightarrow{\mathbf{x}}_{i}\right)=\sum_{i=1}^{N}\left(\log |\mathbf{W}|+\sum_{j=1}^{n} \log p_{s}\left(\overrightarrow{\mathbf{w}}_{j}^{T} \overrightarrow{\mathbf{x}}_{i}\right)\right)
$$
根据最大似然准则，可以采用梯度下降法求解$\mathcal{L}$的最大值。其中：根据矩阵微积分有：$\nabla_{\mathbf{w}}|\mathbf{W}|=|\mathbf{W}|\left(\mathbf{W}^{-1}\right)^{T}$。则有：
$$
\nabla_{\mathbf{W}} \mathcal{L}=\left[\begin{array}{c}{1-2 g\left(\overrightarrow{\mathbf{w}}_{1}^{T} \overrightarrow{\mathbf{x}}_{i}\right)} \\ {1-2 g\left(\overrightarrow{\mathbf{w}}_{2}^{T} \overrightarrow{\mathbf{x}}_{i}\right)} \\ {\vdots} \\ {1-2 g\left(\overrightarrow{\mathbf{w}}_{n}^{T} \overrightarrow{\mathbf{x}}_{i}\right)}\end{array}\right] \overrightarrow{\mathbf{x}}_{i}^{T}+\left(\mathbf{W}^{-1}\right)^{T}
$$
当迭代求解出$\mathbf{W}$之后，通过$\overrightarrow{\mathbf{s}}=\mathbf{W} \overrightarrow{\mathbf{x}}$。 还原出原始信号。最大似然估计时，假设$\vec{\mathbf{x}}_i$和$\vec{\mathbf{x}}_j$之间是相互独立的。事实上对于语音信号或者其他具有时间连续性依赖性的数据，这个假设不能成立。但是当数据足够多，假设独立对于效果影响不大。如果事先打乱样本，则会加快梯度下降法的收敛速度。