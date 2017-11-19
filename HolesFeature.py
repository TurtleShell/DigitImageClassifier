#HolesFeature
"""
The purpose of this file is to create an input
feature that counts the number and positions
of holes in the numbers. By holes I mean
the inside of circles.
"""


from BreadthFirst import *



def expandNumber(squareMatrix, darkCoordsList):

	fatSquareMatrix = squareMatrix.copy()

	for darkCoords in darkCoordsList:
		surroundingCoordsList = createSurroundCoordsList(darkCoords)
		for surroundingCoords in surroundingCoordsList:
			if ((0 <= surroundingCoords.x < 28) and (0 <= surroundingCoords.y < 28)):
				fatSquareMatrix[surroundingCoords.y, surroundingCoords.x] = 1

	return fatSquareMatrix


def listContainsEdge(coordsList, edgeBounds):

	for coords in coordsList:
		if (coords.isOnEdge()):
			return True

		if (coords.isNumEdgeCoords(edgeBounds)):
			return True

	return False


def validWhitePixel(coords, squareMatrix, edgeBounds):
	if not (coords.isWithinNumEdgeCoords(edgeBounds)):
		return False

	if not (coords.isDark(squareMatrix)):
		return True
	else:
		return False


def isValidNewWhiteCoords(coords, toTraverseList, traversedList, newTraversedList, squareMatrix, edgeBounds):

	if not (validWhitePixel(coords, squareMatrix, edgeBounds)):
		return False

	if ((coords in toTraverseList) or (coords in traversedList) or (coords in newTraversedList)):
		return False

	return True


def expandWhitePixel(coords, toTraverseList, traversedList, newTraversedList, squareMatrix, edgeBounds):

	surroundCoordsList = createSurroundCoordsList(coords)

	for possNewCoords in surroundCoordsList:

		if (isValidNewWhiteCoords(possNewCoords, toTraverseList, traversedList, newTraversedList, squareMatrix, edgeBounds)):
			toTraverseList.append(possNewCoords)

	return False


#returns an empty list if there isn't a hole
def holeAtCoords(squareMatrix, firstCoords, traversedList, edgeBounds):

	toTraverseList = []
	newTraversedList = []

	if not (isValidNewWhiteCoords(firstCoords, toTraverseList, traversedList, newTraversedList, squareMatrix, edgeBounds)):
		return []


	toTraverseList.append(firstCoords)

	while (len(toTraverseList) > 0):
		nextCoords = toTraverseList.pop(0)

		expandWhitePixel(nextCoords, toTraverseList, traversedList, newTraversedList, squareMatrix, edgeBounds)
		newTraversedList.append(nextCoords)

	traversedList.extend(newTraversedList)



	if (listContainsEdge(newTraversedList, edgeBounds)):
		return []
	else:
		return newTraversedList




#hole object:
#[1, size, avgX, avgY]
def findHoles(imgVector):

	holesList = []

	squareMatrix = imgVectorToSquareMatrix(imgVector)
	imgCoords = getCoordsOfNumber(imgVector)
	edgeBounds = getBoundsOfNumber(imgCoords)

	numOfHoles = 0

	traversedList = []
	totalHoleCoords = []


	for y in range(26):
		for x in range(26):
			coords = Coords(x+1,y+1)
			if not (coords.isWithinNumEdgeCoords(edgeBounds)):
				continue

			holeCoords = holeAtCoords(squareMatrix, coords, traversedList, edgeBounds)


			if (len(holeCoords) > 0):
				totalHoleCoords.extend(holeCoords)

				avgCoords = getAverageCoords(holeCoords)
				cmpX, cmpY = compareAvgCoordsToImage(avgCoords, imgVector)

				holeObj = HoleObj(holeCoords, cmpX, cmpY)
				holesList.append(holeObj)


	traversedList = totalHoleCoords
	expSqMatrix = expandNumber(squareMatrix, imgCoords)


	incompleteHolesList = []

	for y in range(26):
		for x in range(26):
			coords = Coords(x+1,y+1)

			if not (coords.isWithinNumEdgeCoords(edgeBounds)):
				continue

			holeCoords = holeAtCoords(expSqMatrix, coords, traversedList, edgeBounds)

			if (len(holeCoords) > 0):

				avgCoords = getAverageCoords(holeCoords)
				cmpX, cmpY = compareAvgCoordsToImage(avgCoords, imgVector)

				holeObj = HoleObj(holeCoords, cmpX, cmpY)

				incompleteHolesList.append(holeObj)

	return holesList, incompleteHolesList


def createHolesFeatureMatrixFromInput(inputMatrix):

	holeObjSize = 4
	holesToConsier = 2
	incompleteHolesToConsier = 2

	featurelessInputMatrix = inputMatrix[0:784,:]

	vectors = np.shape(featurelessInputMatrix)[1]

	inputSize = holeObjSize*(holesToConsier + incompleteHolesToConsier)
	featureMatrix = np.zeros((inputSize, vectors))

	for vectori in range(vectors):
		imgVector = featurelessInputMatrix[:,vectori]
		holesList, incompleteHolesList = findHoles(imgVector)
		
		for i,holeObj in enumerate(holesList[:holesToConsier]):
			fIndex = i*holeObjSize
			featureMatrix[fIndex, vectori] = 1
			featureMatrix[fIndex+1, vectori] = holeObj.numPixelsVal
			featureMatrix[fIndex+2, vectori] = holeObj.cmpX
			featureMatrix[fIndex+3, vectori] = holeObj.cmpY


		for i,incompleteHoleObj in enumerate(incompleteHolesList[:incompleteHolesToConsier]):
			fIndex = (holesToConsier*holeObjSize)+(i*holeObjSize)
			featureMatrix[fIndex, vectori] = 1
			featureMatrix[fIndex+1, vectori] = incompleteHoleObj.numPixelsVal
			featureMatrix[fIndex+2, vectori] = incompleteHoleObj.cmpX
			featureMatrix[fIndex+3, vectori] = incompleteHoleObj.cmpY



	return featureMatrix
