#DataFormatFunctions

import numpy as np
from PIL import Image

from HelperClasses import *
from ValueDefinitions import *


def createMNISTVector(filename):
	image = Image.open(filename).convert("L") #Convert to greyscale
	width, height = image.size
	assert (width==IMG_WIDTH and height==IMG_HEIGHT)
	imgData = image.getdata()

	imageVector = np.zeros((IMG_PIXELS, 1))

	for pixeli in range(len(imgData)):
		pixelValue = (255-imgData[pixeli])/255
		pixelValue = pixelValue - .001
		imageVector[pixeli] = pixelValue

	return imageVector


def createImageVectorFromList(coordsList):
	imageVector = np.zeros((IMG_PIXELS, 1))

	for coords in coordsList:
		drawn_index = coords.getMNISTIndex()
		imageVector[drawn_index] = 1

	return imageVector


def imgVectorToSquareMatrix(imgVector):

	resultMatrix = np.zeros((IMG_HEIGHT,IMG_WIDTH))

	for i in range(IMG_WIDTH):
		flatIndex = i*IMG_HEIGHT
		resultMatrix[i,:] = imgVector[flatIndex:flatIndex+IMG_HEIGHT]

	return resultMatrix


def squareMatrixToImgVector(squareMatrix):

	result = np.zeros((IMG_PIXELS,1))

	i = 0

	for y in range(IMG_HEIGHT):
		for x in range(IMG_WIDTH):
			result[i,0] = squareMatrix[y,x]
			i += 1

	return result


def printMNISTVectorAsVectorInt(imageVector):

	print()

	i = 0

	for yi in range(IMG_HEIGHT):
		curList = []
		for xi in range(IMG_WIDTH):
			val = int(imageVector[i])
			curList.append(val)
			i += 1
		print(curList)

	print()


def getBoundsOfNumber(coordsList):
	minX, minY = IMG_WIDTH+1, IMG_HEIGHT+1
	maxX, maxY = -1, -1

	for coords in coordsList:
		curX, curY = coords.x, coords.y
		if (curX < minX):
			minX = curX

		if (curY < minY):
			minY = curY

		if (curX > maxX):
			maxX = curX

		if (curY > maxY):
			maxY = curY

	return EdgeBounds(minX, minY, maxX, maxY)


def createReadableOutputVector(probVector):
	result = ""

	for i,val in enumerate(probVector):
		#entryStr = str(i) + ": " + str(round(val, 3)) + ", "
		entryStr = str(round(val, 2)) + ", "
		result += entryStr

	return result

