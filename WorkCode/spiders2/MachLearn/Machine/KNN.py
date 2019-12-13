import numpy as np
import operator
import os
'''
KNN优点: 精度高、对异常值不敏感、无输入数据假定
缺点: 计算复杂度高、空间复杂度高
步骤: 收集数据、准备数据: 距离计算所需要的数值、分析数据、
训练算法、测试算法、使用算法。
'''
# 准备数据
def createDataSet( ):
    group = np.array([[1., 1.1], [1., 1.], [0, 0], [0, 0.1]])
    labels = list('AABB')
    return group, labels
# KNN 算法, 用于分类
def classify0(inx, dataSet, labels, k):
 # 检验数据是否符合要求
    if not isinstance(type(inx), np.ndarray):
        try:
           inx = np.array(inx)
        except:
            raise TypeError('can not convert to array type')
    if not isinstance(type(dataSet), np.ndarray):
        try:
            dataSet = np.array(dataSet)
        except:
            raise TypeError('can not convert to array type')
    #计算已知类别数据集的点与当前点之间的距离
    distances = np.sqrt(((inx - dataSet)**2).sum(axis = 1))
    # 按照距离递增次序排序
    sorteddistIndicies = distances.argsort()
    classcount = dict()
    # 选取与当前点距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sorteddistIndicies[i]]
        # 确定前k个点所在类别的出现频率
        classcount[voteIlabel] = classcount.get(voteIlabel, 0) +1
    # 返回前k个点出现频率最高的类别作为当前点的预测分类
    sortedClassCount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

'''实际使用算法'''
# 准备数据: 从文本文件中解析数据
def file2Maxtrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    for index, line in enumerate(arrayOLines):
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat, classLabelVector
# 准备数据: 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet- minVals
    normDataSet = normDataSet/ranges
    return normDataSet, ranges, minVals

# 测试数据: 作为完整程序验证分类器
def datingClassTest(filename):
    horatio = 0.1
    datingDataMat, datingLabels = file2Maxtrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*horatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        if classifierResult != datingLabels[i]:
            errorCount += 1
    return errorCount/float(numTestVecs)

# 手写数字识别
def img_to_vector(filename):
    returnVec = np.zeros((1,32 * 32))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec

def imgs_to_vector(path):
    lists = os.listdir(path)
    datingMat = np.zeros((len(lists), 32*32))
    labels = []
    if not path.endswith('/'):
        path = path+'/'
    for index, each in enumerate(lists):
        labels.append(int(each.split('.')[0].split('_')[0]))
        datingMat[index, :] = img_to_vector(path + each)
    return datingMat, labels

def hand_writing_class_test(path):
    datingMat, labels = imgs_to_vector(path)
    testpath = os.path.dirname(path)
    testpath  = testpath +'/testDigits'
    datingTest, testLabels = imgs_to_vector(testpath)
    returnLabels = [0] * len(datingTest)
    for index, each in enumerate(datingTest):
        returnLabels[index] = classify0(each, datingMat, labels, 3)
    return returnLabels
path = 'C:/pdf/machinelearninginaction/Ch02/trainingDigits'
print(hand_writing_class_test(path))



