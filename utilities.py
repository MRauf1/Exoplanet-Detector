import csv
import numpy as np

NUM_TRAIN = 5087
NUM_FLUXES = 3197

def prepare_data(num_data, filename):

	#X_train = np.array()
	X_train = np.random.rand(num_data, NUM_FLUXES)
	Y_train = np.random.rand(num_data, 1)

	i = 0

	with open(filename) as file:

		csv_reader = csv.reader(file)
		
		for row in csv_reader:
			
			if(i != 0 and i <= 500):
				
				X_train[i - 1] = row[1:]
				Y_train[i - 1] = 0 if row[0] == "1" else 1
				
			i += 1
			

	X_train = X_train.astype(np.float)
	Y_train = Y_train.astype(np.float)
	
	
	return X_train, Y_train


#prepare_data(570, "data/exoTest.csv")