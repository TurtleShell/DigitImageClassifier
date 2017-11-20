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



def assignFeatures(mnistFile, trainSubset, validSusbet, selectedFeaturesObj):

	training_data, validation_data, test_data = loadMINSTVectorSubset(mnistFile, trainSubset, validSusbet, 0)

	featureTrainMatrix = createFeatures(training_data[0], selectedFeaturesObj)
	featureValidMatrix = createFeatures(validation_data[0], selectedFeaturesObj)

	training_data = (featureTrainMatrix, training_data[1])
	validation_data = (featureValidMatrix, validation_data[1])

	return training_data, validation_data


def createFeatures(inputMatrix, sfo, silent=False):

	if (sfo.breadthFeature):
		if not silent: print("Creating breadth feature")
		brInput = createBreadthFeatureMatrixFromInput(inputMatrix, BF_IDEAL_CHUNK_NUM)

	if (sfo.holesFeature):
		if not silent: print("Creating holes feature")
		holesInput = createHolesFeatureMatrixFromInput(inputMatrix)

	if (sfo.lineLenFeature):
		if not silent: print("Creating line length feature")
		llInput = createLongestLinesFeatureMatrixFromInput(inputMatrix)

	#====Combine Features====#
	if (sfo.inputFeature):
		result = inputMatrix

	if (sfo.breadthFeature):
		if (result is None):
			result = brInput
		else:
			result = np.concatenate((result, brInput),0)

	if (sfo.holesFeature):
		if (result is None):
			result = holesInput
		else:
			result = np.concatenate((result, holesInput),0)

	if(sfo.lineLenFeature):
		if (result is None):
			result = llInput
		else:
			result = np.concatenate((result, llInput),0)

	return result

