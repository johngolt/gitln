import numpy as np
# 单层决策树分类器
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''数据矩阵、划分的特征、阈值、阈值不等式为:大于还是小于'''
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray # 返回为1, -1

def buildStump(dataArr, classLabels, D):
    '''数据矩阵、数据标签、权重'''
    m, n = dataArr.shape
    classLabels = np.array(classLabels)
    numSteps, bestStump, bestClasEst = 10., {}, np.zeros((m,1))
    minError = np.inf
    for i in range(n): # 寻找合适的特征
        # 寻找合适的阈值
        rangeMin = dataArr[:, i].min()
        rangeMax = dataArr[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin +float(j) * stepSize)
                predictedVals = stumpClassify(dataArr, i, threshVal, inequal)
                errArr = np.ones((m,1))
                errArr[predictedVals == classLabels[:, np.newaxis]] = 0
                weightedError = np.dot(D.T, errArr)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
'''
不同的分类器通过串行训练而获得的，每个新分类器都根据已训练出的
分类器的性能来进行训练。boosting是通过集中关注被已有分类器错分的
那些数据来获得新的分类器。
'''
def adaBoostTrainDs(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    classLabels = np.array(classLabels)
    m = dataArr.shape[0]
    D = np.ones((m, 1))/m
    aggClassEst = np.zeros((m,1))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * np.log((1. - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # 如果classLabels == classEst返回-alpha, 反之返回alpha
        # 更新下一个分类器的样本权重
        expon = np.multiply(-1 * alpha * classLabels[:, np.newaxis], classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        # 确定boosting分类器的错误率。
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != classLabels[:, np.newaxis],
                           np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        if errorRate == 0.:
            break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
    m = datToClass.shape[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(datToClass, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)





def loadSimpData():
    dataMat = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1., 1., -1., -1., 1.]
    return dataMat, classLabels
dataMat, classLabels = loadSimpData()
D = np.ones((5, 1))/5
result = adaBoostTrainDs(dataMat, classLabels, 9)
#print(result)
print(buildStump(dataMat, classLabels, D))
