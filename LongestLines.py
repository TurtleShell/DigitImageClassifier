#LongestLines

"""
This file creates a feature that tracks
the longest dark line and the number of
distinct lines in each major direction
"""

import math

from DataFormatFunctions import *
from TraversalHelperFunctions import *
from HelperClasses import *


def getLineLenFromCoords(coords, sqMatrix, direction, traversedCoords):
	nextCoords = coords
	pixelsTraversed = 0

	while (nextCoords.isValid() and nextCoords.isDark(sqMatrix)):
		traversedCoords.append(nextCoords)
		pixelsTraversed += 1

		nextCoords = Coords(nextCoords.x+direction[0], nextCoords.y+direction[1])

	return pixelsTraversed


def getLineObjForDir(sqMatrix, direction):
	traversedCoords = []
	lineLenPointObjList = []
	lineLens = []
	countedCoords = []

	for y in range(IMG_HEIGHT):
		for x in range(IMG_WIDTH):
			coords = Coords(x, y)
			if (coords not in traversedCoords):
				lineLen = getLineLenFromCoords(coords, sqMatrix, direction, traversedCoords)
				lineLenPointObj = LineLenPointObj(coords, lineLen)
				lineLenPointObjList.append(lineLenPointObj)
				lineLens.append(lineLen)

	maxLen = max(lineLens)

	for lineLenPointObj in lineLenPointObjList:
		lineLen = lineLenPointObj.length
		if (lineLen >= LL_LEN_THRESH):
			lineLenCoords = lineLenPointObj.coords
			if (coordsDistantFromList(lineLenCoords, countedCoords, LL_COORD_DIST_THRESH)):
				countedCoords.append(lineLenCoords)

	linesOverThresh = len(countedCoords)

	return LineLenDirectionObj(maxLen, linesOverThresh)



def getLongestLinesObject(imgVector):
	sqMatrix = imgVectorToSquareMatrix(imgVector)

	llList = []

	for direction in DIRECTIONS_LIST:
		lineObj = getLineObjForDir(sqMatrix, direction)
		llList.append(lineObj)

	return llList


def setLongestLinesFeature(imgVector, vectori, featureMatrix):
	llList = getLongestLinesObject(imgVector)

	for i,lineObj in enumerate(llList):
		fIndex = i*2
		featureMatrix[fIndex,vectori] = lineObj.maxLenVal
		featureMatrix[fIndex+1,vectori] = lineObj.linesVal


def createLongestLinesFeatureMatrixFromInput(inputMatrix):

	vectors = np.shape(inputMatrix)[1]
	featureMatrix = np.zeros((LL_INPUT_SIZE, vectors))

	for vectori in range(vectors):
		imgVector = inputMatrix[:,vectori]
		setLongestLinesFeature(imgVector, vectori, featureMatrix)

	return featureMatrix

