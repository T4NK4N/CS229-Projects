import numpy as np
import matplotlib.pyplot as plt


def loadDataset(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        temp = [1.0]
        for i in range(len(lineArr)-1):
            temp.extend([float(lineArr[i])])
        dataMat.append(temp)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def LR(dataMat,labelMat):
    alpha = 0.01
    numFeature = len(dataMat[0])
    num = len(dataMat)
    initTheta = []
    for i in range(numFeature):
        initTheta.extend([0.01])
    theta = np.array(initTheta).reshape(numFeature, 1)
    temp = theta
    x = np.array(dataMat).reshape(num, numFeature)
    y = np.array(labelMat).reshape(num, 1)

    x_mat = []
    for i in range(numFeature):
        x_mat.append(np.array(dataMat)[:, i].reshape(num, 1))
    for i in range(500):
        for j in range(numFeature):
            temp[j] = theta[j] + alpha * np.sum((y - sigmoid(np.dot(x, theta))) * x_mat[j])
        theta = temp
    return theta


def testLR(theta,testDataMat,testLabelMat):
    errorCount = 0.0
    num = len(testDataMat)
    numFeature = len(testDataMat[0])
    x = np.array(testDataMat).reshape(num, numFeature)
    y = np.array(testLabelMat).reshape(num, 1)
    for i in range(num):
        if sigmoid(np.dot(x[i],theta)) < 0.5 and y[i] == 1.0:
            errorCount+=1
        elif sigmoid(np.dot(x[i],theta)) >= 0.5 and y[i] == 0.0:
            errorCount+=1
    return errorCount/num



if __name__ == '__main__':
    dataMat, labelMat = loadDataset('testSet.txt')
    theta = LR(dataMat, labelMat)
    # dataArr = np.array(dataMat)
    # n = np.shape(dataArr)[0]
    # xcord1 = []
    # xcord2 = []
    # ycord1 = []
    # ycord2 = []
    # for i in range(n):
    #     if int(labelMat[i]) == 1:
    #         xcord1.append(dataArr[i, 1])
    #         ycord1.append(dataArr[i, 2])
    #     else:
    #         xcord2.append(dataArr[i, 1])
    #         ycord2.append(dataArr[i, 2])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xcord1, ycord1, s=30, c='red')
    # ax.scatter(xcord2, ycord2, s=30, c='green')
    # x = range(-4, 4)
    # y = (-theta[0]-theta[1]*x)/theta[2]
    # ax.plot(x,y)
    # plt.show()
    testDataMat,testLabelMat = loadDataset('testSet.txt')
    print(testDataMat)
    print(testLabelMat)

    print(theta)
    errorrate = testLR(theta,testDataMat,testLabelMat)
    print(errorrate)


