#BasicFunctions

"""
The purpose of this file is to give
users simple functions to access
this project
"""

import sys
import time
import datetime

from VectorNeuralNetwork import *
from Features import *
from NumberClassifyGui import *



def trainAndTestDigitClassifier(
	mnistFile, saveNetworkName=None,
	trainSubset=None, validSubset=None,
	dimensions=None, weightRange=None,
	iterations=None, learningRate=None, reg=None,
	batchSize=None, momentumDecay=None,
	inputFeature=None, breadthFeature=None,
	holesFeature=None, lineLenFeature=None):

	#====Input====#
	if not (trainSubset): trainSubset = 50
	if not (validSubset): validSusbet = 10

	print("Train Subset Size:", trainSubset)
	print("Test subset size:", validSusbet)
	#Note: I'm testing using the validation subset. I don't want to touch the test
	#data until the project absolutley finished

	#=====Features=====#
	if not (inputFeature): 		inputFeature 	= True
	if not (breadthFeature):	breadthFeature 	= True
	if not (holesFeature): 		holesFeature 	= True
	if not (lineLenFeature): 	lineLenFeature	= True

	selectedFeaturesObj = SelectedFeatures(inputFeature, breadthFeature,
											holesFeature, lineLenFeature)

	training_data, validation_data = assignFeatures(mnistFile, trainSubset, 
													validSusbet, selectedFeaturesObj)

	inputSize = np.shape(training_data[0])[0]


	#====Training Parameters====#
	dimensions = [inputSize, 240, 10]
	print("Dimensions:", dimensions)

	if not (weightRange):	weightRange = .2
	if not (iterations): 	iterations = 200
	print("Iterations:", iterations)

	if not (learningRate):	learningRate = .01
	if not (reg):			reg = 0
	if not (batchSize):		batchSize = 5
	if not (momentumDecay): momentumDecay = 0

	networkTrainParams = NetworkTrainParams(learningRate, reg, batchSize, momentumDecay)

	neuralNet = VNeuralNet(dimensions, weightRange, selectedFeaturesObj)

	if (lineLenFeature):
		neuralNet.passUpNodes(inputSize-8, 8, 2)

	startTime = time.clock()

	neuralNet.trainNetwork(training_data[0], training_data[1], networkTrainParams, iterations)

	newTime = time.clock()
	programTime = round(newTime - startTime, 2)
	print()
	print("Train Time:", programTime, "seconds")

	testNeuralNet(neuralNet, validation_data)

	if (saveNetworkName):
		saveNeuralNet(neuralNet, saveNetworkName)



def main():

	if (len(sys.argv) < 2):
		print("Please enter directory of the MNIST data")
		return

	mnistFile = sys.argv[1]

	if (len(sys.argv) >= 3):
		saveNetworkName = sys.argv[2]
	else:
		saveNetworkName="UnnamedNetwork"

	trainAndTestDigitClassifier(mnistFile, saveNetworkName)

	runGui(saveNetworkName)

if __name__ == "__main__":
	main()

