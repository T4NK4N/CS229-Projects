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

def GDA(dataMat, labelMat):
    dataArr = np.array(dataMat)
    u0 = np.zeros(3)
    u1 = np.zeros(3)
    u0num = 0.0
    u1num = 0.0
    u0count = np.zeros(3)
    u1count = np.zeros(3)
    for i in range(len(labelMat)):
        if labelMat[i] == 0:
            u0count += dataArr[i]
            u0num += 1.0
        else:
            u1count += dataArr[i]
            u1num += 1.0
    for i in range(len(u0count)):
        u0[i] = u0count[i]/u0num
        u1[i] = u1count[i]/u1num
    point = np.array([(u0[1]+u1[1])/2, (u0[2]+u1[2])/2])
    k = 1.0/((u1[2]-u0[2])/(u1[1]-u0[1]))
    return point, k


if __name__ == '__main__':
    dataMat, labelMat = loadDataset('testSet.txt')

    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    point, k = GDA(dataMat, labelMat)
    print(point, k)
    x = range(-4, 4)
    y = k*(x-point[0])+point[1]
    ax.plot(x, y)
    plt.show()
