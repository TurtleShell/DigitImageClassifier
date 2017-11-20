#Vector Neural Network

"""
This file contains the neural network itself, including
functions to train, save, and load a network.
"""

import numpy as np
import math
import pickle
import random
import sys

from MNISTLoader import *


class VNeuralNet(object):
	def __init__(self, dimensions, weightRange, selectedFeaturesObj):
		self.dimensions = dimensions
		self.layers = len(self.dimensions)
		self.momentumMatrices = []
		self.selectedFeaturesObj = selectedFeaturesObj
		for i in range(self.layers):
			self.momentumMatrices.append(None)

		self.initializeWeights(weightRange)


	def getBotLayerDim(self):
		return self.dimensions[0]

	def getFirstHidLayerOutput(self):
		return self.outputMList[1]

	def getSelectedFeaturesObj(self):
		return self.selectedFeaturesObj

	def initializeWeights(self, valueRange):
		lowerbound = -valueRange/2
		upperbound = valueRange/2

		self.weightsList = []

		for layeri in range(self.layers):
			if (layeri == 0):
				dummyMatrix = np.zeros((1,1))
				self.weightsList.append(dummyMatrix)
			else:
				prevLayerNus = self.dimensions[layeri-1]
				thisLayerNus = self.dimensions[layeri]
				weightMatrix = np.random.uniform(lowerbound, upperbound, (thisLayerNus, prevLayerNus))

				self.weightsList.append(weightMatrix)


	#If there's an important input feature that should more directly
	#impact the output, use this function to isolate input nodes
	def passUpNodes(self, nodeIndexStart, nodeIndexSize, groupSize):
		layer1WM = self.weightsList[1]
		i = 0
		groupi = 0
		while (i < nodeIndexSize):
			layer1WM[groupi,:] = 0

			for inputNode in range(groupSize):
				layer1WM[groupi, nodeIndexStart+i] = 1
				i += 1

			groupi += 1


	#inputMatrix is a (d, N) sized martrix
	def feedForward(self, inputMatrix):

		inputMShape = np.shape(inputMatrix)

		self.signalMList = []
		self.outputMList = []


		#overall goal for each layer is to get its output:
		#get outputs from previous layer
		#use weights at current layer to get signals
		#turn signals into outputs for this layer
		for layeri in range(self.layers):

			if (layeri == 0):
				self.outputMList.append(inputMatrix)
			else:
				prevOMatrix = self.outputMList[layeri-1].astype(np.float32)
				weightMatrix = self.weightsList[layeri].astype(np.float32)
				signalMatrix = np.dot(weightMatrix, prevOMatrix)
				self.signalMList.append(signalMatrix)

				#Now we need to turn the signal into an output
				#Each element is the element from the signal matrix through the nonlinearity
				#vectorizedNonlin = np.vectorize(evalNonlinearity)
				outputMatrix = vectorizedNonlin(signalMatrix)
	
				self.outputMList.append(outputMatrix)

		return self.outputMList[-1]



	#def backPropogate(self, inputMatrix, targetMatrix, learningRate, reg, momentumDecay):
	def backPropogate(self, inputMatrix, targetMatrix, trainParams):

		self.feedForward(inputMatrix)

		#Each delta matrix is layersize X N
		deltaMList = []
		for i in range(self.layers):
			deltaMList.append("DummyDeltaMatrix")

		for i in range(self.layers - 1):
			layeri = self.layers-1-i


			#Compute the delta for all cases, they get summed up when the gradient is computed
			if (layeri == self.layers-1):
				#For final layer, each element of delta matrix = (deriv of cost fn)*(deriv of nonlin)
				#derivCostMatrix is sizeOfOutputLayer X N
				#derivCostMatrix is a combination of targetMatrix and self.outputMList[-1]
				netOutputMatrix = self.outputMList[-1]
				derivCostMatrix = (netOutputMatrix - targetMatrix)
				derivNonlinMatrix = vectorizedDerivNonlin(netOutputMatrix)
				deltaMatrix = derivCostMatrix
				deltaMList[layeri] = deltaMatrix


			else:
				aboveWeightMatrix = self.weightsList[layeri+1]
				aboveDeltaMatrix = deltaMList[layeri+1]


				aboveWeightMatrix = aboveWeightMatrix.astype(np.float32)
				aboveDeltaMatrix = aboveDeltaMatrix.astype(np.float32)
				summedWDMatrix = np.dot(np.transpose(aboveWeightMatrix), aboveDeltaMatrix)

				layerOutputMatrix = self.outputMList[layeri]
				derivNonlinMatrix = vectorizedDerivNonlin(layerOutputMatrix)
				deltaMatrix = np.multiply(summedWDMatrix, derivNonlinMatrix)
				deltaMList[layeri] = deltaMatrix


			belowOutputMatrix = self.outputMList[layeri-1]
			belowOutputMatrix = belowOutputMatrix.astype(np.float32)
			deltaMatrix = deltaMatrix.astype(np.float32)
			gradientMatrix = np.transpose(np.dot(belowOutputMatrix, np.transpose(deltaMatrix)))
			gradientMatrix = trainParams.learningRate*gradientMatrix
			regMatrix = trainParams.reg * trainParams.learningRate * self.weightsList[layeri]
			momentumMatrix = self.applyMomentum(layeri, gradientMatrix, trainParams.momentumDecay)

			self.weightsList[layeri] = self.weightsList[layeri] - gradientMatrix - regMatrix



	#returns the gradient matrix scaled by momentum
	def applyMomentum(self, layeri, gradientMatrix, momentumDecay):
		if (momentumDecay == 0):
			return -1*gradientMatrix

		momentumMatrix = self.momentumMatrices[layeri]

		if (momentumMatrix is None):
			self.momentumMatrices[layeri] = -1*gradientMatrix
			return -1*gradientMatrix

		"""decay"""
		momentumMatrix = momentumDecay*momentumMatrix

		"""create updated gradient matrix"""
		result = (-1*gradientMatrix) + momentumMatrix 

		"""update momentum matrix"""
		self.momentumMatrices[layeri] = result
		return result


	def trainNetwork(self, train_input, train_labels, trainParams, iterations):

		inputCases = np.shape(train_input)[1]
		print()

		for epoch in range(iterations):

			seed = random.randint(0, 50000)

			np.random.seed(seed)
			train_inputT = np.transpose(train_input)
			np.random.shuffle(train_inputT)
			train_input = np.transpose(train_inputT)

			np.random.seed(seed)
			train_labelsT = np.transpose(train_labels)
			np.random.shuffle(train_labelsT)
			train_labels = np.transpose(train_labelsT)

			i = 0
			while (i < inputCases):

				percentComplete = 100*((epoch + (i/inputCases))/iterations)
				printProgress(percentComplete)


				input_batch = train_input[:, i : i+trainParams.batchSize]
				label_batch = train_labels[:, i : i+trainParams.batchSize]
		
				self.backPropogate(input_batch, label_batch, trainParams)

				i += trainParams.batchSize

		sys.stdout.flush()


def saveNeuralNet(neuralNet, fileName):
	f = open(fileName, 'wb')
	pickle.dump(neuralNet, f)
	f.close()


def loadNeuralNet(fileName):
	try:
		f = open(fileName, 'rb')
		neuralNet = pickle.load(f)
		f.close()
		return neuralNet
	except:
		print("ERROR\nNetwork named "+fileName+" not found.")
		exit()


def evalNonlinearity(value):
	#Can swap the returned function for a different lonlinearity
	return logisticNonLin(value)


def logisticNonLin(value):
	try:
		result = 1/(1+math.exp(-1*value))
		return result
	except OverflowError:
		return 0

vectorizedNonlin = np.vectorize(evalNonlinearity)


def derivLogistic(output):
		return output*(1-output)

vectorizedDerivNonlin = np.vectorize(derivLogistic)


def printProgress(percentComplete):
	percentString = "\rProgress: %d" % percentComplete
	percentString += "% "

	sys.stdout.flush()
	sys.stdout.write(percentString)
