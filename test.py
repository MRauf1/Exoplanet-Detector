import sys
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from utilities import prepare_data, get_positives_and_negatives, get_confusion_matrix, get_precision, get_recall, get_F1
import numpy as np


NUM_TEST = 570
NUM_FLUXES = 3197


#Test the model either by predicting or evaluating test images
def test(predict_or_evaluate, model_path):

	#Different results due to loading the model - model is compiled exactly, meaning Dropout still remains
	model = load_model(model_path)

	X_test, Y_test = prepare_data(NUM_TEST, "data/exoTest.csv")

	#Reshape the array from 2D into 3D
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

	#Predict the inputted images' output
	predictions = model.predict(X_test, verbose = 1)
	print("Predictions: \n" + str(predictions))

	if(predict_or_evaluate == "--evaluate"):

		#Evaluate the model on test images
		evaluations = model.evaluate(X_test, Y_test)
		print("Loss: " + str(evaluations[0]))
		print("Accuracy: " + str(evaluations[1] * 100) + " %")

		true_positives, false_positives, true_negatives, false_negatives = get_positives_and_negatives(predictions, Y_test)

		print("True positive: " + str(true_positives))
		print("False positive: " + str(false_positives))
		print("True negative: " + str(true_negatives))
		print("False negative: " + str(false_negatives))

		confusion_matrix = get_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives)
		print("Confusion Matrix:\n" + str(confusion_matrix[0]) + "\n" + str(confusion_matrix[1]))

		precision = get_precision(true_positives, false_positives)
		print("Precision: " + str(precision))

		recall = get_recall(true_positives, false_negatives)
		print("Recall: " + str(recall))

		F1_score = get_F1(precision, recall)
		print("F1 Score: " + str(F1_score))


#Code for running the program from the terminal
terminal_length =  len(sys.argv)

if(terminal_length >= 2):

	#Help command
	if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):

		print("--predict for predicting an output for a set of testing input")
		print("\tSpecify the name of the model (without file extension) to predict with")
		print("--evaluate for evaluating the model on a set of testing input and their true outputs")
		print("\tSpecify the name of the model (without file extension) to evaluate with")

	#Predict command
	elif((sys.argv[1] == "--predict" or sys.argv[1] == "--evaluate") and terminal_length == 3):

		model_path = "models/" + sys.argv[2] + ".h5"
		predict_or_evaluate = sys.argv[1]

		if(os.path.isfile(model_path)):

			print("Predicting images with the model called " + sys.argv[2])
			test(predict_or_evaluate, model_path)

		else:

			print("File does not exist. Make sure the name is typed correctly and without the file extension")

	#Predict command with no model_name
	elif((sys.argv[1] == "--predict" or sys.argv[1] == "--evaluate") and terminal_length == 2):

		print("Please enter the model name as an argument after --predict/--evaluate (without file extension)")

	#Invalid command
	else:

		print("Invalid command.")
		print("Use -h or --help for the list of all possible commands")

else:

	print("Not enough arguments given")
	print("Use -h or --help for the list of all possible commands")
