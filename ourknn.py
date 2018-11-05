import math
import operator

def euclideanDistance(test, train, length):
	distance = 0
	for x in range(length):
		distance += pow((test[x] - train[x]), 2)
	return math.sqrt(distance)

def getMjVotes(neighbors, y):
	classVotes = {}
	nhbrs = y[neighbors]
	for x in range(len(nhbrs)):
		response = nhbrs[x]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def myKnn(testdata, traindata, y, k):
	responses = []
	for X1 in testdata:
		distances = []
		position = 0
		for X2 in traindata:
			dist = euclideanDistance(X1, X2, len(X1))
			distances.append((position, dist))
			position += 1
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		response = getMjVotes(neighbors,y)
		responses.append(response)
	return responses
