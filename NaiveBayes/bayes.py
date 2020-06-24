import numpy as np
import re
# 生成data和label
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec

# 把所有文档所有单词装入一个list
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 文档词袋模型
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary" % word)
    return returnVec

# 计算文档中每个词属于0或者1的概率
def trainNB0(trainMat, trainCategory):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return pAbusive, p1Vect, p0Vect

# 对文章进行分类，vec2Classify为待分类的文章向量
def classifyNB(vec2Classify, p1Vect, p0Vect, pAbusive):
    p0 = sum(vec2Classify*p0Vect)+np.log(pAbusive)
    p1 = sum(vec2Classify*p1Vect)+np.log(pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

# 对文章test进行分类
def testingNB(test):
    listOPosts, listClasses = loadDataSet()
    vocablist = createVocabList(listOPosts)
    trainMat = []
    for postingDoc in listOPosts:
        trainMat.append(bagOfWord2Vec(vocablist, postingDoc))
    pAbusive, p1Vect, p0Vect = trainNB0(trainMat, listClasses)
    thisdoc = np.array(bagOfWord2Vec(vocablist, test))
    return classifyNB(thisdoc, p1Vect, p0Vect, pAbusive)

# 切分文本
def textParse(text):
    listOfTokens = re.split(r'\W+', text)
    return [word.lower() for word in listOfTokens if len(word) > 0]

def spamTest():
    docList = []
    classList =[]
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    dataSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        testSet.append(dataSet[randIndex])
        del(dataSet[randIndex])
    trainMat = []
    trainClasses = []
    for doc in dataSet:
        trainMat.append(bagOfWord2Vec(vocabList, docList[doc]))
        trainClasses.append(classList[doc])
    pAbusive, p1Vect, p0Vect = trainNB0(trainMat, trainClasses)
    errorCount = 0.0
    for i in testSet:
        vec2Classify = bagOfWord2Vec(vocabList, docList[i])
        if classifyNB(vec2Classify, p1Vect, p0Vect, pAbusive) != classList[i]:
            errorCount += 1
    return errorCount/10



if __name__ == '__main__':
    test1 = ['love', 'my', 'dalmation']
    test2 = ['stupid', 'garbage']
    print(testingNB(test1))
    errorRate = 0.0
    for i in range(10):
        errorRate += spamTest()
    print(errorRate/10)




