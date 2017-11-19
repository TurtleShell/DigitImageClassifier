#Features

"""
The purpose of this file
is to transform input matricies
into feature matricies
"""

from ValueDefinitions import *
from MNISTLoader import *
from BreadthFirst import *
from HolesFeature import *
from LongestLines import *
from TestFunctions import *



def assignFeatures(trainSubset, validSusbet, inputFeature, breadthFeature, holesFeature, lineLenFeature):

	training_data, validation_data, test_data = loadMINSTVectorSubset(trainSubset, validSusbet, 0)

	featureTrainMatrix = createFeatures(training_data[0], inputFeature, breadthFeature, holesFeature, lineLenFeature)
	featureValidMatrix = createFeatures(validation_data[0], inputFeature, breadthFeature, holesFeature, lineLenFeature)

	training_data = (featureTrainMatrix, training_data[1])
	validation_data = (featureValidMatrix, validation_data[1])

	return training_data, validation_data


def createFeatures(inputMatrix, inputFeature, breadthFeature, holesFeature, lineLenFeature):

	if (breadthFeature):
		print("Creating breadth feature")
		brInput = createBreadthFeatureMatrixFromInput(inputMatrix, BF_IDEAL_CHUNK_NUM)

	if (holesFeature):
		print("Creating holes feature")
		holesInput = createHolesFeatureMatrixFromInput(inputMatrix)

	if (lineLenFeature):
		print("Creating line length feature")
		llInput = createLongestLinesFeatureMatrixFromInput(inputMatrix)

	#====Combine Features====#
	if (inputFeature):
		result = inputMatrix

	if (breadthFeature):
		if (result is None):
			result = brInput
		else:
			result = np.concatenate((result, brInput),0)

	if (holesFeature):
		if (result is None):
			result = holesInput
		else:
			result = np.concatenate((result, holesInput),0)

	if(lineLenFeature):
		if (result is None):
			result = llInput
		else:
			result = np.concatenate((result, llInput),0)

	return result

