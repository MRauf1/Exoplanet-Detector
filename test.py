import sys
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from utilities import prepare_data
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
	if(predict_or_evaluate == "--predict"):

		predictions = model.predict(X_test, verbose = 1)
		print(predictions)
		for i in range(NUM_TEST):
			if(predictions[i] == 1):
				print(predictions[i])



	else:

		#Evaluate the model on test images
		evaluations = model.evaluate(X_test, Y_test)
		print(evaluations)



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
