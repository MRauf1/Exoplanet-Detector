import sys
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, CuDNNGRU, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from imblearn.over_sampling import SMOTE
from utilities import prepare_data
import numpy as np




NUM_TRAIN = 5087
NUM_FLUXES = 3197
EPOCHS = 10



#Create the architecture of the model as well as compile it
def create_model():

	model = Sequential()

	#Convolutional Layer
	model.add(Conv1D(filters = 64, kernel_size = 8, strides = 4, input_shape = (NUM_FLUXES, 1)))
	model.add(MaxPooling1D(pool_size = 4, strides = 2))
	model.add(Activation('relu'))

	#GRU Layer
	model.add(CuDNNGRU(units = 256, return_sequences = True))

	#Flatten 3D data into 2D format
	model.add(Flatten())

	#Fully Connected Layer
	model.add(Dense(units = 16, activation = "relu"))
	model.add(Dropout(rate = 0.5))
	model.add(BatchNormalization())

	#Final Activation Layer
	model.add(Dense(units = 1, activation = "sigmoid"))

	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

	print(model.summary())

	return model


#Train the model with training and validation images using data augmentation or without it
def train(model_name):

	#SMOTE for upsampling the minority class
	X_train, Y_train = prepare_data(NUM_TRAIN, "data/exoTrain.csv")
	sm = SMOTE()
	X_train, Y_train = sm.fit_sample(X_train, Y_train)

	#Reshape the array from 2D into 3D
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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
