#BreadthFirst

import numpy as np
import math

from DataFormatFunctions import *
from TraversalHelperFunctions import *
from HelperClasses import *





def setDirectionVector(coordsList, caseIndex, featureMatrix, avgCoords, numOfGroupsList):

	if (len(coordsList) == 0):
		return

	lastCoords = avgCoords

	for i, coords in enumerate(coordsList):
		#xChange, yChange = getDirectionBtCoords(avgCoords, coords)
		xChange, yChange = getDirectionBtCoords(lastCoords, coords)
		featureMatrix[i*3,caseIndex] = xChange
		featureMatrix[(i*3)+1,caseIndex] = yChange

		featureMatrix[(i*3)+2,caseIndex] = numOfGroupsList[i]-1

		lastCoords = coords




def createAvgCoordsList(coordsList, chunks):

	chunkSize = len(coordsList)/chunks

	avgCoordsList = []
	numOfGroupsList = []

	startIndex = 0
	endIndex = int(chunkSize)
	i = 1


	while (startIndex+chunkSize <= len(coordsList)):

		if (endIndex+chunkSize >= len(coordsList)):
			curChunk = coordsList[startIndex:]
		else:
			curChunk = coordsList[startIndex:endIndex]

		curAvgCoords = getAverageCoords(curChunk)
		avgCoordsList.append(curAvgCoords)

		numOfGroupsList.append(howManyGroups(curChunk))

		startIndex = endIndex

		i += 1
		endIndex = int(i*chunkSize)

	if (len(avgCoordsList) != chunks):
		print("coords list:", len(coordsList))
		print("avg coords list:", len(avgCoordsList))

	return avgCoordsList, numOfGroupsList




#def findFirstCoords(squareMatrix):
def findFirstCoords(imgVector):

	squareMatrix = imgVectorToSquareMatrix(imgVector)

	foundCoords = None

	for y in range(28):
		for x in range(28):
			curCoords = Coords(x,y)
			if (curCoords.isDark(squareMatrix)):
				foundCoords = curCoords
				break
		if (foundCoords):
			break

	if (foundCoords is None):
		return


	traversedList = traverseNumber(imgVector, foundCoords, False)
	bottomCoords = traversedList[-1]

	reverseTraverseList = traverseNumber(imgVector, bottomCoords, False)
	trueFirstCoords = reverseTraverseList[-1]

	return trueFirstCoords



def isValidNewCoords(coords, toTraverseList, traversedList, squareMatrix):

	if not (coords.isValid() and coords.isDark(squareMatrix)):
		return False

	if ((coords in toTraverseList) or (coords in traversedList)):
		return False

	return True



def expandPixel(coords, toTraverseList, traversedList, squareMatrix):

	surroundCoordsList = createSurroundCoordsList(coords)

	for possNewCoords in surroundCoordsList:
		if (isValidNewCoords(possNewCoords, toTraverseList, traversedList, squareMatrix)):
			toTraverseList.append(possNewCoords)
			#print(possNewCoords)
		#else:
		#	print(possNewCoords)



def traverseNumber(imgVector, firstCoords, expandLocal):
	if (firstCoords is None):
		return []

	squareMatrix = imgVectorToSquareMatrix(imgVector)

	toTraverseList = []
	traversedList = []

	toTraverseList.append(firstCoords)

	if not (expandLocal):
	
		while (len(toTraverseList) > 0):
			nextCoords = toTraverseList.pop(0)
			expandPixel(nextCoords, toTraverseList, traversedList, squareMatrix)
			traversedList.append(nextCoords)

	else:


		nextCoords = toTraverseList.pop(0)
		expandPixel(nextCoords, toTraverseList, traversedList, squareMatrix)
		traversedList.append(nextCoords)
		lastCoords = nextCoords

		traverseLaterList = []
	
		while (len(toTraverseList) > 0):
			nextCoords = toTraverseList.pop(0)
	
			if (areCoordsClose(nextCoords, lastCoords)):
				expandPixel(nextCoords, toTraverseList, traversedList, squareMatrix)
				traversedList.append(nextCoords)

				lastCoords = nextCoords
			else:
				traverseLaterList2 = traverseLaterList.copy()
				for coords in traverseLaterList2:
					if (coords in traversedList):
						traverseLaterList.remove(coords)

				if (len(traverseLaterList) == 0) and not (nextCoords in traversedList):
					traverseLaterList.append(nextCoords)

		while (len(traverseLaterList) > 0):
			nextCoords = traverseLaterList.pop(0)
			expandPixel(nextCoords, traverseLaterList, traversedList, squareMatrix)
			traversedList.append(nextCoords)

	return traversedList


def setBreadthFirstFeature(imgVector, chunks, caseIndex, featureMatrix):
	firstCoords = findFirstCoords(imgVector)
	if (firstCoords is None):
		return
	travCoords = traverseNumber(imgVector, firstCoords, True)
	if (len(travCoords) < chunks):
		firstCoords = Coords(14,14)
		travCoords = traverseNumber(imgVector, firstCoords, True)

		if (len(travCoords) < chunks):
			return

	avgCoordsList, numOfGroupsList = createAvgCoordsList(travCoords, chunks)

	avgCoords = getAverageCoords(travCoords)

	#printMNISTVectorAsVectorInt(createImageVectorFromList(avgCoordsList))

	setDirectionVector(avgCoordsList, caseIndex, featureMatrix, avgCoords, numOfGroupsList)


def createBreadthFeatureMatrixFromInput(inputMatrix, chunks):

	#chunks = 2 #delete

	vectors = np.shape(inputMatrix)[1]
	featureMatrix = np.zeros(((chunks*3), vectors))

	for vectori in range(vectors):
		imgVector = inputMatrix[:,vectori]
		setBreadthFirstFeature(imgVector, chunks, vectori, featureMatrix)

	#print(featureMatrix)

	return featureMatrix


def getCoordsOfNumber(imgVector):

	firstCoords = findFirstCoords(imgVector)
	travCoords = traverseNumber(imgVector, firstCoords, True)
	if (len(travCoords) < 20):
		firstCoords = Coords(14,14)
		travCoords = traverseNumber(imgVector, firstCoords, True)

	return travCoords


def getAvgCoordsOfImage(imgVector):
	travCoords = getCoordsOfNumber(imgVector)
	avgCoords = getAverageCoords(travCoords)
	return avgCoords


def compareAvgCoordsToImage(coords, imgVector):
	imgAvgCoords = getAvgCoordsOfImage(imgVector)
	return getDirectionBtCoords(coords, imgAvgCoords)



def main():
	

	#coordsList = [(0,0), (2,2), (1,-1),(2,3)]
#
	#print(howManyGroups(coordsList))

	#return



	trainingCases = 20
	validCases = 2


	#trainingData, validationData, testData = loadMINSTVectorSubset(trainingCases, validCases, 1)
#
	#trainInputMatrix = trainingData[0]
#
	#createBreadthFeatureMatrixFromInput(trainInputMatrix)
#
	#return


	#coords = (0,0)
	#print(createSurroundCoordsList(coords))
	#return



	myImageVector = createMNISTVector("BreadthImg.bmp")[:,0]
	#print(createBreadthFeatureMatrixFromInput(myImageVector))
	#return


	firstCoords = findFirstCoords(myImageVector)
	resultCoords = traverseNumber(myImageVector, firstCoords, True)

	print("len rc", len(resultCoords))

	#chunks = len(resultCoords)/10
	chunks = 10
	avgCoordsList = createAvgCoordsList(resultCoords, chunks)


	#resultImgVector = createImageVectorFromList(avgCoordsList)
	#printMNISTVector(resultImgVector)


	#resultVector = createDirectionVector(avgCoordsList)
	#print(resultVector)


	#featureMatrix = np.zeros(((chunks*2)+2, 2))
	#setDirectionVector(avgCoordsList, 0, featureMatrix)
	#setDirectionVector(avgCoordsList, 1, featureMatrix)
#
	#print(featureMatrix)

	#resultCoords1 = resultCoords[150:]
	##print(resultCoords)
	#resultImgVector = createImageVectorFromList(resultCoords1)
	#printMNISTVector(resultImgVector)
	

	resultCoords1 = resultCoords[0:30]
	#print(resultCoords)
	resultImgVector = createImageVectorFromList(resultCoords1)
	printMNISTVector(resultImgVector)

	resultCoords2 = resultCoords[30:60]
	resultImgVector = createImageVectorFromList(resultCoords2)
	printMNISTVector(resultImgVector)

	resultCoords3 = resultCoords[60:90]
	resultImgVector = createImageVectorFromList(resultCoords3)
	printMNISTVector(resultImgVector)

	resultCoords4 = resultCoords[90:120]
	resultImgVector = createImageVectorFromList(resultCoords4)
	printMNISTVector(resultImgVector)

	resultCoords5 = resultCoords[120:150]
	resultImgVector = createImageVectorFromList(resultCoords5)
	printMNISTVector(resultImgVector)

	resultCoords6 = resultCoords[150:]
	resultImgVector = createImageVectorFromList(resultCoords6)
	printMNISTVector(resultImgVector)


if __name__ == "__main__":
	main()
