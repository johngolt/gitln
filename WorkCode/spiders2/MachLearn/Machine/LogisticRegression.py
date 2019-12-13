import numpy as np
'''
处理缺失值
使用可用特征的均值来填充缺失值
使用特殊值来填充缺失值
忽略有缺失值的样本
使用相似样本的均值添补缺失值
使用机器学习算法预测缺失值
'''
def sigmoid(x):
    return 1/(1+np.exp(-x))
# 梯度下降算法
def gradAscent(dataMatIn, classLabels):
    labelMat = classLabels[:, np.newaxis]
    m, n = np.shape(dataMatIn)
    maxCycles = 500
    weights = np.ones((n,1))
    alpha = 0.001
    for k in range(maxCycles):
        h = sigmoid(np.dot(labelMat, weights))
        error = labelMat - h
        weights += alpha * dataMatIn.T * error
    return weights

# SGD 算法
def stocGradAscent(dataMatrix, classLabels):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid((dataMatrix[i] * weights).sum())
        error = classLabels[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights

def classifyVector(inx, weights):
    prob = sigmoid(np.dot(inx, weights))
    if prob > 0.5:
        return 1
    else:
        return 0


