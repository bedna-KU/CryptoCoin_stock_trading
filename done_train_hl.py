#!/usr/bin/env python3
from keras.callbacks import ModelCheckpoint, LambdaCallback
import numpy as np
import os
import sys
import csv
from numpy.core.defchararray import encode

from numpy.lib.function_base import vectorize
# Own library
from models.model import lstm_hl

from datetime import datetime

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

####################################################
# PARAMETERS
####################################################
INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 25
BATCH_SIZE = 128
FILEPATH = "weights_hl.hdf5"
####################################################

# Load data
def data_load ():
	data = {}

	# Read CSV file into array
	data = []
	with open ("file_new.csv", newline = "") as csvfile:
		reader = csv.reader (csvfile, delimiter = ',')
		for row in reader:
			data.append (row)

	data_rows_count = len(data)
	print (">>> data rows count:", data_rows_count)

	doge_max = 0
	doge_min = 999999999
	for row in data:
		if float(row[2]) > doge_max:
			doge_max = float(row[2])
		if float(row[3]) < doge_min:
			doge_min = float(row[3])
	print (">>> DOGE min", doge_min)
	print (">>> DOGE max", doge_max)

	with open('min_max_doge.csv', 'w') as f:
		write = csv.writer(f, delimiter=',')
		csv_out = [doge_min, doge_max]
		write.writerow(csv_out)
		print("Save min max")

	# Get data from columns
	data_time_doge = get_column(data, 0)
	data_high_doge = get_column(data, 2)
	data_low_doge = get_column(data, 3)
	data_close_doge = get_column(data, 4)

	input_high_doge_arr = []
	input_low_doge_arr = []
	output_close_doge_arr = []
	loop = 0
	yes = True
	while yes:
		input_high_doge = data_high_doge[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		input_low_doge = data_low_doge[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		output_close_doge = data_close_doge[INPUT_LEN + SHIFT * loop : INPUT_LEN + OUTPUT_LEN + SHIFT * loop]
		if len(input_high_doge) < INPUT_LEN or len(output_close_doge) < OUTPUT_LEN:
			yes = False
		else:
			input_high_doge_arr.append(input_high_doge)
			input_low_doge_arr.append(input_low_doge)
			output_close_doge_arr.append(output_close_doge)
			loop += SHIFT

	input_high_doge_arr = np.array(input_high_doge_arr)
	input_low_doge_arr = np.array(input_low_doge_arr)
	print(">>> count list", len(input_high_doge_arr))

	encode2 = np.vectorize(encode)
	input_high_doge_arr = encode2(input_high_doge_arr, doge_max)
	input_low_doge_arr = encode2(input_low_doge_arr, doge_max)

	X = np.array([input_high_doge_arr[0], input_low_doge_arr[0]])
	for index in range(1, len(input_high_doge_arr)):
		print(index)
		X = np.append(X, [input_high_doge_arr[index], input_low_doge_arr[index]], axis = 0)
	X = np.reshape(X, (len(input_high_doge_arr),2 ,INPUT_LEN))

	print(X)
	y = np.array(output_close_doge_arr, dtype = float)
	y = encode2(y, doge_max)
	print(y)
	return X, y

def get_column(matrix, i):
	return [row[i] for row in matrix]

# Invert encoding
def decode (value, max):
	return value * max

# Encode data
def encode (value, max):
	result = float(value) / max
	return result

# Run every epoch
def on_epoch_end (epoch, logs):
	print (">>>LOGS>>>", logs)

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
	model.fit (X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = callbacks)

model = lstm_hl (INPUT_LEN, OUTPUT_LEN)
model_load (model)
print (model.summary())
X, y = data_load ()
model_train (model, X, y)
