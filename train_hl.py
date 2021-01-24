#!/usr/bin/env python3
from keras.callbacks import ModelCheckpoint, LambdaCallback
import numpy as np
import os
import sys
import csv

from numpy.lib.function_base import vectorize
# Own library
from models.model import lstm_hl

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

####################################################
# PARAMETERS
####################################################
INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 28
BATCH_SIZE = 128
FILEPATH = "weights_hl.hdf5"
####################################################

# Load data
def data_load ():
	data = []
	# Read CSV file into array
	with open ("file_new.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data.append (row)

	data_rows_count = len(data)
	print (">>> data rows count:", data_rows_count)
	max_high = 0
	min_high = 999999999
	for row in data:
		if float(row[2]) > max_high:
			max_high = float(row[2])
		if float(row[2]) < min_high:
			min_high = float(row[2])
	print (">>> min_high", min_high)
	print (">>> max_high", max_high)

	# Get data from columns (time + ohlcv)
	# data_time = get_column(data, 0)
	# data_open = get_column(data, 1)
	data_high = get_column(data, 2)
	data_low = get_column(data, 3)
	data_close = get_column(data, 4)
	# data_volume = get_column(data, 5)

	input_high_list = []
	input_low_list = []
	output_close_arr = []
	loop = 0
	yes = True
	while yes:
		input_high = data_high[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		input_low = data_low[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		output_close = data_close[INPUT_LEN + SHIFT * loop : INPUT_LEN + OUTPUT_LEN + SHIFT * loop]
		if len(input_high) < INPUT_LEN or len(output_close) < OUTPUT_LEN:
			yes = False
		else:
			input_high_list.append(input_high)
			input_low_list.append(input_low)
			output_close_arr.append(output_close)
			loop += SHIFT

	input_high_arr = np.array(input_high_list)
	input_low_arr = np.array(input_low_list)
	print(">>> count arr", len(input_high_arr))
	print(">>> count arr shape high", input_high_arr.shape)
	print(">>> count arr shape low", input_high_arr.shape)

	# To ndarray
	X = np.array([input_high_arr, input_low_arr], dtype = float)
	X = np.reshape(X, (len(input_high_list), INPUT_LEN, 2))
	y = np.array(output_close_arr, dtype = float)
	encode2 = np.vectorize(encode)
	X = encode2(X, max_high)
	y = encode2(y, max_high)

	with open('min_max_doge.csv', 'w') as f:
		write = csv.writer(f, delimiter=',') 
		csv_out = [min_high, max_high]
		write.writerow(csv_out)
		print("Save min max")

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

# Run evry epoch
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
