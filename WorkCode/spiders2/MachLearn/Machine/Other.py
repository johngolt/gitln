import numpy as np
def pca(dataMat, topNfeat = 9999):
    meanVals = dataMat.mean()
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved)
    eigVals, eigVects = np.linalg.eig(covMat)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat + 1) : -1]
    redEigVects = eigVects[:, eigValInd]
    lowDataDat = np.dot(meanRemoved, redEigVects)
    reconMat = np.dot(lowDataDat, redEigVects.T) + meanVals
    return lowDataDat, reconMat

