#TraversalHelperFunctions

import math

from HelperClasses import *


def getDirectionBtCoords(coords1, coords2):
	xChange = coords2.x - coords1.x
	yChange = coords2.y - coords1.y

	return xChange, yChange


def areCoordsTouching(coords1, coords2):

	xDist = abs(coords2.x - coords1.x)

	if (xDist <= 1):
		yDist = abs(coords2.y - coords1.y)
		if (yDist <= 1):
			return True
	else:
		return False


def areCoordsClose(coords1, coords2):
	threshold = 4

	xDist, yDist = getDirectionBtCoords(coords1, coords2)

	dist = math.sqrt((xDist**2) + (yDist**2))


	if (dist < threshold):
		return True
	else:
		return False


def createSurroundCoordsList(coords):

	surroundCoordsList = []

	for yShift in range(-1, 2):
		for xShift in range(-1, 2):
			if ((yShift == 0) and (xShift == 0)):
				continue

			newCoords = Coords(coords.x + xShift, coords.y + yShift)
			surroundCoordsList.append(newCoords)

	return reversed(surroundCoordsList)


def getAverageCoords(coordsList):

	xTotal = 0
	yTotal = 0

	coordsNum = len(coordsList)

	for coords in coordsList:
		xTotal += coords.x
		yTotal += coords.y

	xResult = xTotal/coordsNum
	yResult = yTotal/coordsNum

	return Coords(int(xResult), int(yResult))


def coordDistance(coords1, coords2):
	xDist = abs(coords2.x - coords1.x)
	yDist = abs(coords2.y - coords1.y)
	return xDist + yDist



def coordsDistantFromList(coords, coordsList, distThreshold):

	for listCoords in coordsList:
		curDist = coordDistance(coords, listCoords)
		if (curDist < distThreshold):
			return False

	return True


def howManyGroups(coordsList):

	if (len(coordsList) == 0):
		return 0

	toTraverse = coordsList.copy()
	traversedList = [toTraverse.pop(0)]

	while (len(traversedList) > 0):
		traversedCoords = traversedList.pop(0)

		i = 0

		while (i < len(toTraverse)):
			curCoords = toTraverse[i]
			if (areCoordsTouching(traversedCoords, curCoords)):
				#foundTouching += 1
				traversedList.append(toTraverse.pop(i))
			else:
				i += 1


	return 1 + howManyGroups(toTraverse)


