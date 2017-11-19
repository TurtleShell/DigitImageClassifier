#TestZone


import sys
import time
import datetime

from VectorNeuralNetwork import *
from MNISTLoader import *
from BreadthFirst import *
from HolesFeature import *
from LongestLines import *
from TestFunctions import *





def main():


	trainSubset = 50
	validSusbet = 10
	testSubset = 0

	threshold = .2
	save = False


	#=====Feature Zone=====#
	justInput = 0
	justFeatures = 0


	shiftFeature = 0

	inputFeature = 1
	breadthFeature = 1
	holesFeature   = 0
	lineLenFeature = 1



	print("Train Subset Size:", trainSubset)

	training_data, validation_data, test_data = loadMINSTVectorSubset(trainSubset, validSusbet, testSubset)


	if (justInput):
		breadthFeature = 0
		holesFeature = 0


	elif (justFeatures):
		breadthFeature = True
		holesFeature = True


	inTrainInput = training_data[0]
	inValidInput = validation_data[0]


	if (breadthFeature):
		print("Creating breadth feature: training data")
		brTrainInput = createBreadthFeatureMatrixFromInput(training_data[0][:784,:], 20)
		print("Creating breadth feature: test data")
		brValidInput = createBreadthFeatureMatrixFromInput(validation_data[0][:784,:], 20)

	if (holesFeature):
		print("Creating holes feature: training data")
		holesTrainInput = createHolesFeatureMatrixFromInput(training_data[0][:784,:])
		print("Creating holes feature: test data")
		holesValidInput = createHolesFeatureMatrixFromInput(validation_data[0][:784,:])


	if (lineLenFeature):
		print("Creating line length feature: training data")
		llTrainInput = createLongestLinesFeatureMatrixFromInput(training_data[0][:784,:])
		print("Creating line length feature: test data")
		llValidInput = createLongestLinesFeatureMatrixFromInput(validation_data[0][:784,:])


	#====Combining Features Zone====#

	if (inputFeature):
		trainInput = inTrainInput
		validInput = inValidInput

	if (breadthFeature):
		if (trainInput is None):
			trainInput = brTrainInput
			validInput = brValidInput
		else:
			trainInput = np.concatenate((trainInput, brTrainInput),0)
			validInput = np.concatenate((validInput, brValidInput),0)

	if (holesFeature):
		if (trainInput is None):
			trainInput = holesTrainInput
			validInput = holesValidInput
		else:
			trainInput = np.concatenate((trainInput, holesTrainInput),0)
			validInput = np.concatenate((validInput, holesValidInput),0)

	if(lineLenFeature):
		if (trainInput is None):
			trainInput = llTrainInput
			validInput = llValidInput
		else:
			trainInput = np.concatenate((trainInput, llTrainInput),0)
			validInput = np.concatenate((validInput, llValidInput),0)

	training_data = (trainInput, training_data[1])
	validation_data = (validInput, validation_data[1])


	print("training data", np.shape(training_data[0]))

	inputSize = np.shape(training_data[0])[0]

	print("inputsize", inputSize)


	#dimensions = [784, 30, 10]
	#dimensions = [inputSize, 100, 10]
	dimensions = [inputSize, 240, 10]
	#dimensions = [inputSize, 8, 10]
	#dimensions = [inputSize, 10]

	print("Dimensions:", dimensions)

	#weightRange = .4
	weightRange = .2

	#iterations = 250
	iterations = 500

	print("Iterations:", iterations)

	learningRate = .01
	#learningRate = .001
	reg = 0
	#batchSize = 10
	batchSize = 5
	momentumDecay = 0
	#nnTrainParams = [learningRate, reg, batchSize, momentumDecay]
	networkTrainParams = NetworkTrainParams(learningRate, reg, batchSize, momentumDecay)

	numOfNetworks = 1
	secondGuess = False

	print("numOfNetworks", numOfNetworks)

	#networkList = createMulti(numOfNetworks, dimensions, weightRange)
	neuralNet = VNeuralNet(dimensions, weightRange)
	neuralNet.passUpNodes(860, 8, 2)

	print("weight matrix")
	print(neuralNet.weightsList[1][:4, 858:])

	startTime = time.clock()

	#trainMulti(networkList, training_data, nnTrainParams, iterations)

	neuralNet.trainNetwork(training_data[0], training_data[1], networkTrainParams, iterations)

	newTime = time.clock()
	programTime = round(newTime - startTime, 2)
	print()
	print("Train Time:", programTime, "seconds")


	testNeuralNet(neuralNet, validation_data)

	#saveNeuralNet(neuralNet, "TestZoneNetworkPassUp")





if __name__ == "__main__":
	main()
