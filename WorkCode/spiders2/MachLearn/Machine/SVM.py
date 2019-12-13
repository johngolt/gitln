import numpy as np

def loadDataSet(filename):
    dataMat, labelMat = [], []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return:
    '''
    m, n = dataMatIn.shape
    b, alphas, iter = 0, np.zeros((m, 1)), 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.dot(np.multiply(alphas, classLabels[:, np.newaxis]).T, np.dot(dataMatIn, dataMatIn[i, :].T))) +b
            Ei = fXi - float(classLabels[i])
            if ((classLabels[i] * Ei < -toler) and (alphas[i] < C)) or ((classLabels[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.dot(np.multiply(alphas, classLabels[:, np.newaxis]).T, np.dot(dataMatIn, dataMatIn[j, :].T))) + b
                Ej = fXj - float(classLabels[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if classLabels[i] != classLabels[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue
                eta = 2. * np.dot(dataMatIn[i, :], dataMatIn[j, :]) - np.dot(dataMatIn[i, :], dataMatIn[i, :]) - np.dot(dataMatIn[j, :], dataMatIn[j, :])
                if eta >= 0:
                    continue
                alphas[j] -= classLabels[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    continue
                alphas[i] += classLabels[j] * classLabels[i] *(alphaJold - alphas[j])
                b1 = b - Ei - classLabels[i] * (alphas[i] - alphaIold) * np.dot(dataMatIn[i, :], dataMatIn[i, :]) \
                - classLabels[j] * (alphas[j] - alphaJold) * np.dot(dataMatIn[i, :], dataMatIn[j, :])
                b2 = b - Ej - classLabels[i] * (alphas[i] - alphaIold) * np.dot(dataMatIn[i, :], dataMatIn[j, :]) \
                - classLabels[j] * (alphas[j] - alphaJold) * np.dot(dataMatIn[j, :], dataMatIn[j, :])
                if alphas[i] > 0 and C > alphas[i]:
                    b = b1
                elif alphas[j] > 0 and C> alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2)/2
                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas

# 完整的platt SMO
class optStruct:
    def __init__(self, dataIn, classLabels, C, toler):
        self.X = dataIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataIn.shape[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.eCache = np.zeros((self.m, 2))

def calcEk(oS, k):
    fXk = float(np.dot(np.multiply(oS.alphas, oS.labelMat[:, np.newaxis]).T, np.dot(oS.X, oS.X[k, :].T))) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK, maxDeltaE, Ej = -1, 0, 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList) >1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK, maxDeltaE, Ej = k, deltaE, Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
                L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
                H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H: return 0
        eta = 2. * np.dot(oS.X[i,:], oS.X[j, :].T) - np.dot(oS.X[i, :], oS.X[i, :].T) - np.dot(oS.X[j, :], oS.X[j, :].T)
        if eta >= 0: return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001: return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * np.dot(oS.X[i, :], oS.X[i, :].T)\
        - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * np.dot(oS.X[i, :], oS.X[j, :].T)
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * np.dot(oS.X[i, :], oS.X[j, :].T)\
        - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * np.dot(oS.X[j, :], oS.X[j, :].T)
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C: oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C: oS.b = b2
        else: oS.b = (b1 + b2)/2
        return 1
    else: return 0

def smo(dataIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(dataIn, classLabels, C, toler)
    iter = 0
    entireSet, alphaPairsChanged = True, 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter +=1
        else:
            nonBoundIs = np.nonzero((oS.alphas > 0) * (oS.alphas < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        if entireSet: entireSet = False
        elif alphaPairsChanged ==0: entireSet = True
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    m, n = dataArr.shape
    W = np.zeros((n, 1))
    for i in range(m):
        W += np.multiply(alphas[i] * classLabels[i], dataArr[i, :].T)
    return W

def kernelTrans(X, A, kTup):
    m, n = X.shape
    K = np.zeros((m, 1))
    if kTup[0] =='lin': K = np.dot(X, A.T)
    elif kTup[0] =='rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = np.dot(deltaRow, deltaRow.T)
        K = np.exp(K/(-1 * kTup[1] ** 2))
    else: raise NameError('That Kernel is not recognized')
    return K
class optStructKernel:
    def __init__(self, dataIn, classLabels, C, toler, kTup):
        self.X = dataIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataIn.shape[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.eCache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)





















filename1 = 'C:/pdf/machinelearninginaction/Ch06/testSet.txt'
dataMat, labelMat = loadDataSet(filename1)
b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
print(b, alphas[alphas > 0])
