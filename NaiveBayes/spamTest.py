import bayes
import numpy as np

def spamTest():
    docList = []
    classList =[]
    fullText = []
    for i in range(1, 26):
        wordList = bayes.textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = bayes.textParse(open('email/ham/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    dataSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        testSet.append(dataSet[randIndex])
        del(dataSet[randIndex])
    trainMat = []
    trainClasses = []
    for doc in dataSet:
        trainMat.append(bayes.bagOfWord2Vec(vocabList, docList[doc]))
        trainClasses.append(classList[doc])
    pAbusive, p1Vect, p0Vect = bayes.trainNB0(trainMat, trainClasses)
    errorCount = 0.0
    for i in testSet:
        vec2Classify = bayes.bagOfWord2Vec(vocabList, docList[i])
        if bayes.classifyNB(vec2Classify, p1Vect, p0Vect, pAbusive) != classList[i]:
            errorCount += 1
    return errorCount/10



if __name__ == '__main__':
    errorRate = 0.0
    for i in range(10):
        errorRate += spamTest()
    print(errorRate/10)