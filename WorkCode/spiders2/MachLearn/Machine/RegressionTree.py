import numpy as np

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append([float(item) for item in curLine])
    return np.array(dataMat)

def binSplitDataSet(dataSet, feature, value):
    if not isinstance(feature, int):
        raise ValueError('must be int')
  # print(np.nonzero(dataSet[:, feature] > value))
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    return dataSet[:, -1].mean()

def regErr(dataSet):
    return dataSet[:, -1].var() * dataSet.shape[0]

def linearSolve(dataSet):
    m, n = dataSet.shape
    X = np.ones_like(dataSet)
    y = np.ones((m, 1))
    X[:, 1:n] = dataSet[:, 0:n-1]
    y = dataSet[:, -1]
    xTx = np.dot(X.T, X)
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular')
    ws = np.dot(np.linalg.inv(xTx), np.dot(X.T, y))
    return ws, X, y

def modelLeaf(dataSet):
    ws, X, y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws, X, y = linearSolve(dataSet)
    yHat = np.dot(X, ws)
    return ((y-yHat)**2).sum()

def chooseBestSplit(dataSet, leaftType = regLeaf, errType = regErr, ops = (1,4)):
    tolS, tolN = ops
    if len(set(dataSet[:, -1].T.tolist())) == 1:
        return None, leaftType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitValue in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitValue)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitValue
                bestS = newS
    if (S - bestS) < tolS:
        return None, leaftType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        return None, errType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regErr, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 回归树剪枝
def isTree(obj):
    return type(obj).__name__ == 'dict'

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left'])/2

def prune(tree, testData):
    if testData.shape[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNomerge = ((lSet[:, - 1] - tree['left'])**2).sum() + ((rSet[:, -1] - tree['right'])**2).sum()
        treeMean = getMean(tree)
        errorMerge = ((testData[:, -1] - treeMean)**2).sum()
        if errorMerge < errorNomerge:
            return treeMean
        else:
            return tree
    else:
          return tree

# 回归树预测
def regTreeEval(model, inData):
    return float(model)
def modelTreeEval(model, inData):
    n = inData.shape[1]
    X = np.ones((1, n+1))
    X[:, 1:n+1] = inData
    return float(np.dot(X, model))

def treeForecast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = testData.shape[0]
    yHat = np.zeros((m,1))
    for i in range(m):
        yHat[i] = treeForecast(tree, testData[i], modelEval)
    return yHat






filename1 = 'C:/pdf/machinelearninginaction/Ch09/ex2test.txt'
filename2 = 'C:/pdf/machinelearninginaction/Ch09/ex2.txt'
dataSet = loadDataSet(filename2)
dataSetTest = loadDataSet(filename1)
tree = createTree(dataSet)
tree2 = prune(tree, dataSetTest)

