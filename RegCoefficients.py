import math
import numpy as np
from scipy import linalg

def calculateRegCoef(indVar, depVar):
    xTxI = linalg.inv(np.dot(indVar.T, indVar))
    xTy = np.dot(indVar.T, depVar)
    weights = np.dot(xTxI, xTy)
    return np.dot(indVar, weights)


def calculateLWLRCoef(indVar, depVar, testPoint, k):
    numItems = np.shape(indVar)[0]
    weightMatrix = np.eye(numItems)
    for j in range(numItems):
        dowDistance = math.fabs(testPoint[2] - indVar[j,2])
        taxiDistance = math.fabs(testPoint[1] - indVar[j,1])
        distance = np.linalg.norm([dowDistance, taxiDistance])
        weightMatrix[j,j] = math.exp((-1.0 * distance) / (2.0 * k)**2)
    xTxI = linalg.inv(np.dot(indVar.T, np.dot(weightMatrix, indVar)))
    weights = np.dot(xTxI, np.dot(indVar.T, np.dot(weightMatrix, depVar)))
    return weights


def predictLWLR(indVar, depVar, testData, k):
    numItems = np.shape(testData)[0]
    zHat = np.zeros(numItems)
    for i in range(numItems):
        weights = calculateLWLRCoef(indVar, depVar, testData[i], k)
        zHat[i] = np.dot(testData[i], weights)
    return zHat
