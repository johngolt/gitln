import numpy as np
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat, labelMat = [], []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegression(xArr, yArr):
    xArr = np.array(xArr)
    yArr = np.array(yArr)
    xTx = np.dot(xArr.T, xArr)
    if np.linalg.det(xTx) == 0:
        return
    ws = np.dot(np.linalg.inv(xTx), np.dot(xArr.T, yArr[:, np.newaxis]))
    return ws
filename = 'C:/pdf/machinelearninginaction/Ch08/ex0.txt'
xArr, yArr = loadDataSet(filename)
print(standRegression(xArr, yArr))

# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k = 1.):
    xArr, yArr = np.array(xArr), np.array(yArr)
    m = xArr.shape[0]
    weights = np.eye(m)
    for j in range(m):
        diffMat = testPoint - xArr[j, :]
        weights[j, j] = np.exp(np.dot(diffMat, diffMat[:, np.newaxis])/(-2.*k**2))
    xTx = np.dot(np.dot(xArr.T, weights), xArr)
    if np.linalg.det(xTx) == 0:
        return
    ws = np.dot(np.linalg.inv(xTx), np.dot(xArr.T, np.dot(weights, yArr[:, np.newaxis])))
    return np.dot(testPoint, ws)
def lwlrTest(testArr, xArr, yArr, k = 1.):
    testArr = np.array(testArr)
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

yHat = lwlrTest(xArr, xArr, yArr, 0.003)
print(yHat[:5])

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = np.dot(xMat.T, xMat)
    denom = xTx + np.eye(xTx.shape[0]) * lam
    if np.linalg.det(denom) == 0:
        return
    ws = np.dot(np.linalg.inv(denom), np.dot(xMat.T, yMat[:, np.newaxis]))
    return ws

def regressionError(yArr, yHat):
    return ((yArr - yHat)**2).sum()

# 前向逐步线性回归
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xArr, yArr = np.array(xArr), np.array(yArr)
    yMean = yArr.mean()
    yArr = yArr - yMean
    xMean = xArr.mean()
    xVar = xArr.var()
    xArr = (xArr - xMean)/xVar
    m, n  = xArr.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps *sign
                yTest = np.dot(xArr, wsTest[:, np.newaxis])
                rssE = regressionError(yArr, yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws[:, np.newaxis]
    return returnMat

