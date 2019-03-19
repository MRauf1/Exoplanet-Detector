import csv
import numpy as np

NUM_FLUXES = 3197

def prepare_data(num_data, filename):

	#X_train = np.array()
	X_train = np.random.rand(num_data, NUM_FLUXES)
	Y_train = np.random.rand(num_data, 1)

	i = 0

	with open(filename) as file:

		csv_reader = csv.reader(file)

		for row in csv_reader:

			if(i != 0 and i <= num_data):

				X_train[i - 1] = row[1:]
				Y_train[i - 1] = 0 if row[0] == "1" else 1

			i += 1


	X_train = X_train.astype(np.float)
	Y_train = Y_train.astype(np.float)


	return X_train, Y_train



#Returns the positives and negatives predicted by the model
def get_positives_and_negatives(array, true_output):

	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0

	for i in range(len(array)):
		if(array[i] < 0.5):
			if(true_output[i] == 0):
				true_negatives += 1
			else:
				false_negatives += 1
		else:
			if(true_output[i] == 1):
				true_positives += 1
			else:
				false_positives += 1


	return true_positives, false_positives, true_negatives, false_negatives



def get_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives):

	return ([[true_positives, false_positives], [false_negatives, true_negatives]])


def get_precision(true_positives, false_positives):

	return (true_positives / (true_positives + false_positives))


def get_recall(true_positives, false_negatives):

	return (true_positives / (true_positives + false_negatives))


def get_specificity(false_positives, true_negatives):

	return (true_negatives / (false_positives + true_negatives))


def get_F1(precision, recall):

	return (2 * ((precision * recall) / (precision + recall)))
