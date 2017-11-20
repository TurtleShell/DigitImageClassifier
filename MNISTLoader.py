#MNIST Loader

import pickle
import gzip
import numpy as np
from PIL import Image


def loadMNISTVector(mnistFile):
	try:
		f = gzip.open(mnistFile, 'rb')
		training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
		f.close()
	except:
		print("ERROR\nFailed to open MNIST data. Please be sure the proper directory "+
			"is specified and the file is in the format of <filename>.pkl.gz")
		exit()


	training_data = replaceLabelsWithVectors(training_data)
	validation_data = replaceLabelsWithVectors(validation_data)
	test_data = replaceLabelsWithVectors(test_data)

	return training_data, validation_data, test_data


def loadMINSTVectorSubset(mnistFile, trainSubSize, validSubSize, testSubSize):
	trainingData, validationData, testData = loadMNISTVector(mnistFile)

	trainInput = trainingData[0]
	validInput = validationData[0]
	testInput = testData[0]


	trainInputSubset = trainInput[0:trainSubSize]
	validInputSubset = validInput[0:validSubSize]
	testInputSubset = testInput[0:testSubSize]


	trainLabels = trainingData[1]
	validLabels = validationData[1]
	testLabels = testData[1]

	trainLabelsSubset = trainLabels[0:trainSubSize]
	validLabelsSubset = validLabels[0:validSubSize]
	testLabelsSubset = testLabels[0:testSubSize]

	trainDataSubset = (np.transpose(trainInputSubset), np.transpose(trainLabelsSubset))
	validDataSubset = (np.transpose(validInputSubset), np.transpose(validLabelsSubset))
	testDataSubset =  (np.transpose(testInputSubset),  np.transpose(testLabelsSubset))


	return trainDataSubset, validDataSubset, testDataSubset



def vectorizeLabel(labelValue):
	vector = np.zeros((1, 10))[0]
	vector[labelValue] = 1
	return vector


def replaceLabelsWithVectors(data):
	labelVector = data[1]
	newMatrix = np.zeros((len(labelVector), 10))	
	for i in range(len(labelVector)):
		label = labelVector[i]
		newMatrix[i] = vectorizeLabel(label)

	return (data[0], newMatrix)


def main():
	loadMNISTVector()


if __name__ == "__main__":
	main()
