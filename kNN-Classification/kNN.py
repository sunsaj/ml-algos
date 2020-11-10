from numpy import * 
import operator

def createDataSet():
	group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	datasize = dataSet.shape[0]
	diffMat = tile(inX, (datasize, 1)) - dataSet
	SqDiffMat = diffMat ** 2
	SumMat = SqDiffMat.sum(axis=1)
	mat = SumMat ** 0.5
	sortMat = mat.argsort()
	countClass = {}
	for i in range(k):
		l =  labels[sortMat[i]]
		countClass[l] = countClass.get(l,0) + 1

	countVector = sorted(countClass.items(), key=operator.itemgetter(1), reverse= True)
	return countVector[0][0]

def file2matrix( filename ):
	love_dictionary = {'didntLike':0, 'smallDoses':1, 'largeDoses':2}
	fr = open(filename)
	noOflines = len(fr.readlines())
	retMat = zeros((noOflines, 3))
	classLabels = []
	index = 0
	fr = open(filename)
	for line in fr.readlines():
		line = line.strip()
		listFromline = line.split('\t')
		retMat[index, :] = listFromline[:3]
		if listFromline[-1].isdigit():
			classLabels.append(int(listFromline[-1]))
		else:
			classLabels.append(love_dictionary[listFromline[-1]])
		index += 1

	return retMat, classLabels

def autoNorm(dataSet):
	minVals = dataSet.min(axis=0)
	maxVals = dataSet.max(axis=0)
	rangeVals = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet / tile(rangeVals, (m,1))
	return normDataSet , rangeVals, minVals


def datingClassTest():
	hoRatio = 0.1
	datingDataTest, labels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataTest)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],labels[numTestVecs:m],3)
		print(f"Predicted result: {classifierResult}, Actual result: {labels[i]}")
		if classifierResult != labels[i]:
			errorCount += 1.0

	print(f"Error rate is {errorCount/float(numTestVecs)}")