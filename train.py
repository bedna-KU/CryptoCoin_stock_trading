#!/usr/bin/env python3
from keras.callbacks import ModelCheckpoint, LambdaCallback
import numpy as np
import os
import sys
import csv
# Own library
from models.model import lstm_medium

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

####################################################
# PARAMETERS
####################################################

MAX_LEN = 7
MAX_VALUE = 49
ACTIVATION = "relu"
EPOCHS = 22000
BATCH_SIZE = 8192
FILEPATH = "weights.hdf5"

####################################################

X_nums = []
y_nums = []

# Load data
def data_load ():
	results = []
	# Read CSV file into array
	with open ("file_new.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			results.append (row)

	max_close = 0
	min_close = 999999999
	for row in results:
		if float(row[1]) > max_close:
			max_close = float(row[1])
		if float(row[1]) < min_close:
			min_close = float(row[1])

	print ("min_close", min_close)
	print ("max_close", max_close)

	# Remove firs n'linea with array
	# del results[0:2214]

	data = results

	print ('*** Lines count', len (data))
	numbers = set (x for l in data for x in l)
	numbers = sorted (numbers)
	# ~ MAX_VALUE = len (numbers)
	print ('*** Total numbers: ', MAX_VALUE)
	# Split data to prev/next numbers
	X = data[ : -1440]
	y = data[1440 : ]
	X_nums = X
	y_nums = y
	# To ndarray
	X = np.array (X, dtype = np.float32)
	y = np.array (y, dtype = np.float32)
	# Encode data
	# X = one_hot_encode (X, MAX_VALUE)
	X = encode (X, max_close)
	print(X)
	y = encode (y, max_close)
	# To ndarray
	X = np.array (X)
	return X, y, X_nums, y_nums

# Unvectorize data
def one_hot_decode (seq, max):
	strings = list ()
	for pattern in seq:
		string = str (np.argmax (pattern))
		strings.append (string)
	return ' '.join (strings)

# Vectorize data
def one_hot_encode (value, max_int):
	max_int = max_int + 1
	value_enc = list ()
	for seq in value:
		pattern = list ()
		for index in seq:
			vector = [0 for _ in range (max_int)]
			vector[index] = 1
			pattern.append (vector)
		value_enc.append (pattern)
	return value_enc

# Invert encoding
def decode (value, max):
	return value * max

# Encode data
def encode (value, max):
	result = value / max
	return result

# Run evry epoch
def on_epoch_end (epoch, logs):
	print ("===X_nums", X_nums[-1])
	prev_nums_arr = np.array (X_nums[-1])
	print (">>>PREV>>>>", prev_nums_arr)
	numbers_enc = one_hot_encode ([prev_nums_arr], MAX_VALUE)
	numbers_enc = np.expand_dims (numbers_enc, axis = 3)
	result = model.predict (numbers_enc, batch_size = BATCH_SIZE, verbose=0)
	predicted_nums = decode (result[0], MAX_VALUE)
	predicted_nums = np.around (predicted_nums)
	real_nums = np.array (y_nums[-1])
	print (">>>REAL>>>>", real_nums)
	predicted_nums = predicted_nums.astype ('int')
	print (">>>TIP>>>>>", predicted_nums)
	real_nums_arr = real_nums.tolist ()
	predicted_nums_arr = predicted_nums.tolist ()
	matches = len ([key for key, val in enumerate (predicted_nums_arr) if val in set (real_nums_arr)])
	print (">>>MATCH>>>", matches)

# Load trained weights
def model_load (model):
	if os.path.exists (FILEPATH):
		model.load_weights (FILEPATH)

# Train model
def model_train (model, X, y):
	checkpoint = ModelCheckpoint (FILEPATH, monitor = 'loss',
								 verbose = 1, save_best_only = True,
								 mode = 'min')

	print_callback = LambdaCallback (on_epoch_end = on_epoch_end)
	callbacks = [print_callback, checkpoint]
	X = np.expand_dims (X, axis = 3)
	model.fit (X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = callbacks)

# model = lstm_medium (MAX_LEN, MAX_VALUE, ACTIVATION)
# model_load (model)
# print (model.summary())
X, y, X_nums, y_nums = data_load ()
exit("END")
model_train (model, X, y)
