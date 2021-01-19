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

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# np.set_printoptions(suppress = True)
# np.set_printoptions(precision = 4,
#                        threshold = 10000,
#                        linewidth = 150)

####################################################
# PARAMETERS
####################################################

INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 28
BATCH_SIZE = 128
FILEPATH = "weights.hdf5"

####################################################

# Load data
def data_load ():
	data = []
	# Read CSV file into array
	with open ("file_new.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data.append (row)

	# data = data[500000 : ]

	data_rows_count = len(data)
	print (">>> data rows count:", data_rows_count)
	max_close = 0
	min_close = 999999999
	for row in data:
		if float(row[1]) > max_close:
			max_close = float(row[1])
		if float(row[1]) < min_close:
			min_close = float(row[1])
	print (">>> min_close", min_close)
	print (">>> max_close", max_close)

	yes = True
	# Get data from columns (time + ohlcv)
	data_time = get_column(data, 0)
	data_open = get_column(data, 1)
	data_high = get_column(data, 2)
	data_low = get_column(data, 3)
	data_close = get_column(data, 4)
	data_volume = get_column(data, 5)
	# data_close = np.array(data_close)
	# print(">>> TST", data_close.shape)
	# encode2 = np.vectorize(encode)
	# data_close = encode2(data_close, max_close)
	loop = 0
	input_high_list = []
	input_low_list = []
	output_close_arr = []
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
	print(">>> count list", len(input_high_list))
	print(">>> count arr", len(input_high_arr))
	print(">>> count arr shape high", input_high_arr.shape)
	print(">>> count arr shape low", input_high_arr.shape)
	print(">>> count arr type", type(input_high_arr))
	# print(">>> arr", input_high_arr)

	# To ndarray
	# input_high_arr = np.array(input_high_arr)
	# input_low_arr = np.array(input_low_arr)
	X = np.array([input_high_arr, input_low_arr], dtype = float)
	X = np.reshape(X, (len(input_high_list), INPUT_LEN, 2))
	# print(X)
	y = np.array(output_close_arr, dtype = float)
	print ("+++ type(y)", type(y))
	encode2 = np.vectorize(encode)
	X = encode2(X, max_close)
	y = encode2(y, max_close)
	print(y)
	return X, y

def get_column(matrix, i):
	return [row[i] for row in matrix]

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
	# print ("value.shape", value.shape)
	# print ("value", value)
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
	# X = np.expand_dims (X, axis = 3)
	# print("ccc", Xh.shape)
	# print("ccc", Xh)
	# X = Xh
	# X = np.vstack((Xh, Xl)).T
	# X = np.insert(X, 1, Xl)
	print ("+++", X.shape)
	print ("+++", type(X))
	print ("+++", type(y))
	model.fit (X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = callbacks)

model = lstm_hl (INPUT_LEN, OUTPUT_LEN)
model_load (model)
print (model.summary())
X, y = data_load ()
# print("xxx", Xh.shape)
# print("xxx", Xl.shape)
# print("xxx", y.shape)
# exit("END")
model_train (model, X, y)


