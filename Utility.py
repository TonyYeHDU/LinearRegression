import datetime

def loadDataSet(filePath):
    global lineArr
    dataList = []

    with open(filePath, 'r+') as f:
        numCols = len(open(filePath).readline().split('\t'))
        for line in f.readlines():
            lineArr = []
            currentLine = line.strip().split('\t')
            for i in range(numCols):
                lineArr.append(currentLine[i])
            dataList.append(lineArr)
    return dataList


def prepareDataSet(dataList):
    indVar, depVar = [], []

    for line in dataList:
        durationDifference = float(line[5]) - float(line[4])
        # Monday = 0 and Sunday = 6.
        dow = datetime.datetime(2004,8,int(line[0])).weekday()
        if durationDifference > 0:
            indVar.append([1, int(line[3]), int(dow)])
            depVar.append(int(durationDifference))
    return indVar, depVar
    

def calculateSEofEstimate(depVarTest, depVarHat, k):
    n = np.shape(depVarHat)[0]
    sigma = 0.0

    for i, j in zip(depVarTest, depVarHat):
        sigma += (float(i) - j) ** 2.0

    sigma /= (n - k)
    return sigma ** (1.0/2.0)
