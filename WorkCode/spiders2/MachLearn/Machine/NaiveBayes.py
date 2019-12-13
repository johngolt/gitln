import numpy as np

def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return vocabSet
# 词集模型：将每个词是否出现作为一个特征
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec
# 词袋模型：将每个词出现的次数作为一个特征
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB(trainMatrix, trainCategory):
    '''文档矩阵、类别标签所对应的向量'''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算出先验概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom, p1Denom = 0., 0.
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            # 每个词出现多少次
            p1Num += trainMatrix[i]
            # 一共出现多少个词
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算不同类别下每个词出现的概率
    p1Vect = np.log(p1Num/ p1Denom)
    # 取对数避免乘法运算带来的上溢或下溢
    p0Vect = np.log(p0Num/ p0Denom)
    return p1Vect, p0Vect, pAbusive

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    elif p1 < p0:
        return 0
    else:
        return np.random.choice([0,1])

# 垃圾邮件分类
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


