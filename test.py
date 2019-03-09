import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from random import shuffle
from utilities import prepare_data
import numpy as np

NUM_TEST = 570
NUM_FLUXES = 3197


#Test the model either by predicting or evaluating test images
def test(predict_or_evaluate = "predict"):

	#Different results due to loading the model - model is compiled exactly, meaning Dropout still remains
	model = load_model("model_train.h5")
	
	X_test, Y_test = prepare_data(NUM_TEST, "data/exoTest.csv")

	#Predict the inputted images' output
	if(predict_or_evaluate == "predict"):

		predictions = model.predict(X_test, verbose = 1)
		print(predictions)
		for i in range(NUM_TEST):
			if(predictions[i] == 1):
				print(predictions[i])


	
	else:

		#Evaluate the model on test images
		evaluations = model.evaluate(X_test, Y_test)
		print(evaluations)


test("evaluate")
