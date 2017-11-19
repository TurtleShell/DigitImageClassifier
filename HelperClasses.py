#HelperClasses

import math

from ValueDefinitions import *

class Coords(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y


	def __str__(self):
		return ("(" + str(self.x) + "," + str(self.y) + ")")


	def __eq__ (self, otherCoords):
		return (self.x == otherCoords.x) and (self.y == otherCoords.y)


	def getMNISTIndex(self):
		return (self.y * IMG_WIDTH) + self.x


	def getScaledDown(self, scale):
		xScaled = int(self.x/scale)
		yScaled = int(self.y/scale)

		return Coords(xScaled, yScaled)


	def isDark(self, squareMatrix):
		pixelValue = squareMatrix[self.y, self.x]

		if (pixelValue >= PIXEL_DARK_THRESH):
			return True
		else:
			return False


	def isOnEdge(self):
		if ((self.x == 0) or (self.x == IMG_WIDTH-1) or
			(self.y == 0) or (self.y == IMG_HEIGHT-1)):
			return True
		else:
			return False


	def isNumEdgeCoords(self, edgeBounds):
		if ((self.x == edgeBounds.minX) or (self.x == edgeBounds.maxX) or
			(self.y == edgeBounds.minY) or (self.y == edgeBounds.maxY)):
			return True
		else:
			return False


	def isWithinNumEdgeCoords(self, edgeBounds):
		if ((edgeBounds.minX <= self.x <= edgeBounds.maxX) and
		    (edgeBounds.minY <= self.y <= edgeBounds.maxY)):
			return True
		else:
			return False


	def isValid(self):
		if ((0 <= self.x < IMG_WIDTH) and (0 <= self.y < IMG_HEIGHT)):
			return True
		else:
			return False



class EdgeBounds(object):
	def __init__(self, minX, minY, maxX, maxY):

		self.minX = minX
		self.minY = minY
		self.maxX = maxX
		self.maxY = maxY
		


class LineLenPointObj(object):

	def __init__(self, coords, length):

		self.coords = coords
		self.length = length
		

class LineLenDirectionObj(object):

	def __init__(self, maxLen, lines):

		#self.maxLen = maxLen
		#self.lines = lines

		self.maxLenVal = ((maxLen-1)**2)/50
		self.linesVal = lines/3


		

class HoleObj(object):
	def __init__(self, holeCoords, cmpX, cmpY):

		numPixels = len(holeCoords)
		self.numPixelsVal = math.log(1+numPixels)

		self.cmpX, self.cmpY = cmpX, cmpY


class NetworkTrainParams(object):

	def __init__(self, learningRate, reg, batchSize, momentumDecay):

		self.learningRate = learningRate
		self.reg = reg
		self.batchSize = batchSize
		self.momentumDecay = momentumDecay
