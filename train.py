import sys
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from imblearn.over_sampling import SMOTE
#import keras_metrics
from random import shuffle
from utilities import prepare_data
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam


NUM_TRAIN = 5087
NUM_FLUXES = 3197
EPOCHS = 200



#Create the architecture of the model as well as compile it
def create_model():

	model = Sequential()

	#Fully Connected Layer 1
	model.add(Dense(16, input_shape = ([NUM_FLUXES]), activation = "relu"))
	model.add(Dropout(0.5))

	#Fully Connected Layer 2
	model.add(Dense(4, activation = "relu"))
	model.add(Dropout(0.5))

	#Final Activation Layer
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) #[keras_metrics.binary_precision(), keras_metrics.binary_recall()]

	print(model.summary())

	return model


#Train the model with training and validation images using data augmentation or without it
def train(model_name):

	#SMOTE for upsampling the minority class
	X_train, Y_train = prepare_data(NUM_TRAIN, "data/exoTrain.csv")
	sm = SMOTE()
	X_train, Y_train = sm.fit_sample(X_train, Y_train)

	model = create_model()

	#Add some checkpoints
	tensorboard = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_graph = True, write_images = True)
	checkpoint_train = ModelCheckpoint(model_path, monitor = "loss", save_best_only = True)
	print("Added checkpoints")

	model.fit(x = X_train, y = Y_train, epochs = EPOCHS,
		callbacks = [tensorboard, checkpoint_train])



#Code for running the program from the terminal
terminal_length =  len(sys.argv)

if(terminal_length >= 2):

	#Help command
	if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):

		print("Write the model name as an argument (without file extension)")

	#Train command with model_name
	else:

		model_path = "models/" + sys.argv[1] + ".h5"
		print("Beginning to train the model with the name " + sys.argv[1])
		train(model_path)

else:

	print("Invalid command.")
	print("Use -h or --help for the list of all possible commands")
