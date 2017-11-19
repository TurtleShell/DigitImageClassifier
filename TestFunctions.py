#TestFunctions

import numpy as np


def printIncorrectDict(incorrectDict):
	for i in range(9):
		if (i in incorrectDict):
			numClass = str(i)
			numIncorrect = str(incorrectDict[i])
			print(numClass + " : " + numIncorrect + " incorrect")


def testNeuralNet(neuralNet, testData):

	total_correct = 0

	testInputs = testData[0]
	test_outputs = testData[1]

	inputDim = np.shape(testInputs)[0]	
	testCases = np.shape(testInputs)[1]

	outputMatrix = neuralNet.feedForward(testInputs)

	incorrectNumbers = {}

	for i in range(testCases):

		outputVector = outputMatrix[:,i]
		outputLabel = np.argmax(outputVector)

		actualVector = test_outputs[:,i]
		actualLabel = np.argmax(actualVector)

		if (outputLabel == actualLabel):
			total_correct += 1
		else:
			if not (actualLabel in incorrectNumbers):
				incorrectNumbers[actualLabel] = 1
			else:
				incorrectNumbers[actualLabel] += 1


	accuracy = 100 * (total_correct/testCases)
	print("Accuracy:", accuracy, "%")
	printIncorrectDict(incorrectNumbers)

	return accuracy