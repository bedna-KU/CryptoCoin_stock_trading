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
EPOCHS = 11
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

		# print("xxx", data[item][1])

	# data = data[500000 : ]

	data_rows_count = {}
	for item in DATA_CAT:
		data_rows_count[item] = len(data[item])
		print (">>> data rows count:", data_rows_count[item])
	# print(data["doge"][510080])
	# print(data["btc"][510080])

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
			exit("Error time")
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
			exit("Error time")
		prev_item = int(item[0])

	max_close = {}
	min_close = {}
	for item in DATA_CAT:
		max_close[item] = 0
		min_close[item] = 999999999
		for row in data[item]:
			if float(row[4]) > max_close[item]:
				max_close[item] = float(row[4])
			if float(row[4]) < min_close[item]:
				min_close[item] = float(row[4])
		print (">>> min_close " + item, min_close[item])
		print (">>> max_close " + item, max_close[item])

	with open('min_max_doge.csv', 'w') as f:
		write = csv.writer(f, delimiter=',')
		for item in DATA_CAT:
			csv_out = [min_close[item], max_close[item]]
			write.writerow(csv_out)
			print("Save min max " + item)

	# Get data from columns
	data_time = {}
	data_close = {}
	for item in DATA_CAT:
		data_time[item] = get_column(data[item], 0)
		data_close[item] = get_column(data[item], 4)

	input_close = {}
	output_close = {}
	input_close_arr = {}
	output_close_arr = {}
	for item in DATA_CAT:
		input_close_arr[item] = []
		output_close_arr[item] = []
		loop = 0
		yes = True
		while yes:
			input_close[item] = data_close[item][0 + SHIFT * loop : INPUT_LEN + SHIFT * loop]
			output_close[item] = data_close[item][INPUT_LEN + SHIFT * loop : INPUT_LEN + OUTPUT_LEN + SHIFT * loop]
			if len(input_close[item]) < INPUT_LEN or len(output_close[item]) < OUTPUT_LEN:
				yes = False
			else:
				# print("xxx", input_close)
				input_close_arr[item].append(input_close[item])
				output_close_arr[item].append(output_close[item])
				loop += SHIFT

	# print("ccc", input_close_arr)

	for item in DATA_CAT:
		input_close_arr[item] = np.array(input_close_arr[item])
		print(">>> count list", len(input_close_arr[item]))
		print(">>> count arr {}".format(item), input_close_arr[item].shape)
		# print(">>> arr", input_high_arr)

	# print(list(input_close_arr)[0])
	# print(input_close_arr[list(input_close_arr)[0]])
	# To ndarray
	# input_high_arr = np.array(input_high_arr)
	# input_low_arr = np.array(input_low_arr)

	encode2 = np.vectorize(encode)
	# input_close_arr["doge"] = encode2(input_close_arr["doge"], max_close["doge"])
	# input_close_arr["btc"] = encode2(input_close_arr["btc"], max_close["btc"])

	# print("input_close_arr[doge]", input_close_arr["doge"])

	# print("LEN", len(input_close_arr["doge"]))
	# X = np.array([])
	# for index in range(len(input_close_arr["doge"])):
	# 	doge_line = input_close_arr["doge"][index]
	# 	btc_line = input_close_arr["btc"][index]
	# 	line = np.array((doge_line, btc_line), dtype=float)
	# 	X[index] = np.append(X, line)
	# 	print("===", line)

	# input_close_arr["doge"] = np.reshape(input_close_arr["doge"], (509, 1440, 1))
	# input_close_arr["btc"] = np.reshape(input_close_arr["btc"], (509, 1440, 1))

	X = np.array((input_close_arr["doge"], input_close_arr["btc"]), dtype=float)
	# print("X", X[0])
	# print("X", X[1])
	print("X", X)
	# print("X shape", X.shape)
	# print("X shape", X.shape)
	X = np.reshape(X, (len(input_close_arr["doge"]), INPUT_LEN, 2))
	# print("Xr", X)
	# print("Xr shape", X.shape)
	# aaa = [[11,12,13,14,15], [21,22,23,24,25], [31,32,33,34,35]]
	# bbb = [[111,112,113,114,115], [211,212,213,214,215], [311,312,313,314,315]]
	# aaa = np.array(aaa)
	# bbb = np.array(bbb)
	# ccc = np.array((aaa, bbb))
	# ccc = np.hstack((aaa, bbb))
	# ccc = np.reshape(ccc, (3,2,5))
	# print("AAA", aaa)
	# print("CCC", ccc)
	# print("CCC 0", ccc[1,2])
	# print("CCC shape", ccc.shape)

	y = np.array(output_close_arr["doge"], dtype = float)

	print ("+++ X shape", X.shape)

	y = encode2(y, max_close["doge"])

	# print("X", X)
	# print("y", y)
	# exit()
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


