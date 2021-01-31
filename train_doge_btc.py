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
# List = name : column
DATA_CAT = {"doge" : 4, "btc" : 4}
####################################################

# Load data
def data_load ():
	data = {}
	for item in DATA_CAT:
		data[item] = []
		# Read CSV file into array
		with open ("data_" + item + ".csv", newline = "") as csvfile:
			reader = csv.reader (csvfile, delimiter = ',')
			for row in reader:
				data[item].append (row)

		print("xxx", data[item][1])

	# data = data[500000 : ]

	data_rows_count = {}
	for item in DATA_CAT:
		data_rows_count[item] = len(data[item])
		print (">>> data rows count:", data_rows_count[item])
	print(data["doge"][510080])
	print(data["btc"][510080])
	
	a = 0
	prev_item = 0
	for item in data["doge"]:
		# print(item[0])
		if int(item[0]) - prev_item == 60 * 1000 or prev_item == 0:
			# print("ccc")
			a = a
		else:
			print("xxx", int(item[0]) - prev_item)
			unixtime = int(item[0]) / 1000
			print(">>>", datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S'))
			# exit("END")
		prev_item = int(item[0])

	print()

	a = 0
	prev_item = 0
	for item in data["btc"]:
		# print(item[0])
		if int(item[0]) - prev_item == 60 * 1000 or prev_item == 0:
			# print("ccc")
			a = a
		else:
			print("ccc", int(item[0]) - prev_item)
			unixtime = int(item[0]) / 1000
			print(">>>", datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S'))
			# exit("END")
		prev_item = int(item[0])
	
	exit("END")

	for item in DATA_CAT:
		max_close_doge = 0
		min_close_doge = 999999999
		for row in data_doge:
			if float(row[1]) > max_close_doge:
				max_close_doge = float(row[1])
			if float(row[1]) < min_close_doge:
				min_close_doge = float(row[1])
		print (">>> min_close", min_close_doge)
		print (">>> max_close", max_close_doge)

	max_close_btc = 0
	min_close_btc = 999999999
	for row in data_btc:
		if float(row[1]) > max_close_btc:
			max_close_btc = float(row[1])
		if float(row[1]) < min_close_btc:
			min_close_btc = float(row[1])
	print (">>> min_close", min_close_btc)
	print (">>> max_close", max_close_btc)

	# Get data from columns
	data_time_btc = get_column(data_btc, 0)
	data_close_btc = get_column(data_btc, 4)
	data_time_doge = get_column(data_doge, 0)
	data_close_doge = get_column(data_doge, 4)

	input_close_doge_arr = []
	input_close_btc_arr = []
	output_close_doge_arr = []
	loop = 0	
	yes = True
	while yes:
		input_close_doge = data_close_doge[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		input_close_btc = data_close_btc[0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
		output_close_doge = data_close_doge[INPUT_LEN + SHIFT * loop : INPUT_LEN + OUTPUT_LEN + SHIFT * loop]
		if len(input_close_doge) < INPUT_LEN or len(output_close_doge) < OUTPUT_LEN:
			yes = False
		else:
			input_close_doge_arr.append(input_close_btc)
			input_close_btc_arr.append(input_close_doge)
			output_close_doge_arr.append(output_close_doge)
			loop += SHIFT

	input_close_doge_arr = np.array(input_close_doge_arr)
	input_close_btc_arr = np.array(input_close_btc_arr)
	print(">>> count list", len(input_close_doge_arr))
	print(">>> count arr doge", input_close_doge_arr.shape)
	print(">>> count arr btc", input_close_btc_arr.shape)
	# print(">>> arr", input_high_arr)

	# To ndarray
	# input_high_arr = np.array(input_high_arr)
	# input_low_arr = np.array(input_low_arr)
	X = np.array([input_close_doge_arr, input_close_btc_arr], dtype = float)
	X = np.reshape(X, (len(input_close_doge_arr), INPUT_LEN, 2))
	# print(X)
	y = np.array(output_close_doge_arr, dtype = float)
	print ("+++ type(y)", type(y))
	encode2 = np.vectorize(encode)
	X = encode2(X, max_close_btc)
	y = encode2(y, max_close_doge)
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


