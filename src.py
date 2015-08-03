import numpy as np
from sklearn.cross_validation import train_test_split

dataLocation = 'filename.txt'
rawData = loadDataSet(dataLocation)
xData, zList = prepareDataSet(rawData)

plotObservedData(xData, zList)

# Convert lists to numpy arrays.  Split into test and training sets.
xArray = np.asarray(xData)
zArray = np.asarray(zList)
xyTrain, xyTest, zTrain, zTest = train_test_split(xArray[:,:], zArray[:], test_size=0.3, random_state=42)

# Regular linear regression
zPredict = calculateRegCoef(xArray, zArray)
qqplotResiduals(zPredict, xArray, zArray)

# Locally weighted linear regression
zPredict = predictLWLR(xyTrain, zTrain, xyTest, 0.9)
qqplotResiduals(zPredict, xyTest, zTest)
plotZDifference(xyTest, zTest, zPredict)
estimateSE = calculateSEofEstimate(zTest, zPredict, len(xyTest)-1)
