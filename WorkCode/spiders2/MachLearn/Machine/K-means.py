import numpy as np

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(item) for item in curLine]
        dataMat.append(fltLine)
    return np.array(dataMat)

def distEclud(vecA, vecB):
    return np.sqrt(((vecA - vecB)**2).sum())

# 产生初始迭代点
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.zeros((k, n))
    for j in range(n):
        minJ = dataSet[:, j].min()
        rangeJ = float(dataSet[:, j].max() - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)
    return centroids

def KMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    # 存放类别和到类别中心的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist, minIndex = np.inf, -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            index = np.nonzero(clusterAssment[:, 0] == cent)[0]
            pstInClust = dataSet[index, :]
            centroids[cent, :] = pstInClust.mean()
            # 得到下一个迭代的初始点
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeans = distEclud):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroid0 = dataSet.mean()
    centList = [centroid0]
    # centArr = np.array(centList)
    for j in range(m):
        clusterAssment[j, 1] = distMeans(centroid0, dataSet[j, :])**2
    while len(centList) < k:
        lowestSSE = np.inf # sum of squared error
        for i in range(len(centList)):
            pstInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0], :]
            centroidMat, splitClustAss = KMeans(pstInCurrCluster, 2, distMeans)
            sseSplit = splitClustAss[:, 1].sum()
            sseNotSplit = clusterAssment[np.nonzero(clusterAssment[:, 0] != i)[0], 1].sum()
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentTosplit = i # 最适合切分的簇
                bestNewCents = centroidMat #切分后的质点
                bestClusterAss = splitClustAss.copy() # 得到的切分点的相关信息: 属于的簇、squared error
                lowestSSE = sseSplit + sseNotSplit
        # 更新簇的分配结果
        bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 1)[0], 0] = len(centList)
        bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentTosplit)[0], :] = bestClusterAss
    return centList, clusterAssment




filename1 = 'C:/pdf/machinelearninginaction/Ch10/testSet.txt'
dataMat = loadDataSet(filename1)
print(biKmeans(dataMat, 4))
