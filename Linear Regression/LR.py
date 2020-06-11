import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeature = len(open(filename).readline().split('\t'))-1
    dataMat = []
    labelMat = []
    for line in open(filename).readlines():
        lineArr = []
        curLine = line.split('\t')
        for i in range(numFeature):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def normalEquation(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    theta = (xMat.T*xMat).I*xMat.T*yMat
    return theta

def batchGradientDescent(xArr, yArr):
    alpha = 0.01
    theta = np.array([0.01,0.01]).reshape(2,1)
    temp = theta
    num = len(xArr)
    x= np.array(xArr).reshape(num,2)
    x0 = np.array(xArr)[:,0].reshape(num,1)
    x1 = np.array(xArr)[:,-1].reshape(num,1)
    y = np.array(yArr).reshape(num,1)
    for i in range(100):
        temp[0] = theta[0]+alpha*np.sum((y-np.dot(x,theta))*x0)
        temp[1] = theta[1]+alpha*np.sum((y-np.dot(x,theta))*x1)
        theta = temp
    return theta


def stochasticGradientDescent(xArr, yArr):
    alpha = 0.01
    theta = np.array([0.01, 0.01]).reshape(2, 1)
    temp = theta
    num = len(xArr)
    x = np.array(xArr).reshape(num, 2)
    x0 = np.array(xArr)[:, 0].reshape(num, 1)
    x1 = np.array(xArr)[:, -1].reshape(num, 1)
    y = np.array(yArr).reshape(num, 1)
    for i in range(num):
        temp[0] = theta[0] + alpha * (y[i] - np.dot(x[i], theta)) * x0[i]
        temp[1] = theta[1] + alpha * (y[i] - np.dot(x[i], theta)) * x1[i]
        theta = temp
    return theta


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = np.zeros(np.shape(yArr))       #easier for plotting
    xCopy = np.mat(xArr)
    xCopy.sort(0)
    for i in range(np.shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy




if __name__ == '__main__':
    xArr,yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    k = 0.5
    theta = batchGradientDescent(xArr,yArr)
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # srtInd = xMat[:, 1].argsort(0)
    # xSort = xMat[srtInd][:, 0, :]
    x = xMat.copy()
    y = x*theta
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[:,1], y)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=5)



    plt.show()
    weight = np.mat(np.eye(200))
    print(weight*xArr)






