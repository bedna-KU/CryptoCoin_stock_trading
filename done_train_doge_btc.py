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
EPOCHS = 70
BATCH_SIZE = 128
FILEPATH = "weights_doge_btc.hdf5"
# List = name : column
DATA_CAT = {"doge" : 4, "btc" : 4}
####################################################

# Load data
def data_load():
	data = {}
	for item in DATA_CAT:
		data[item] = []
		# Read CSV file into array
		with open("data_" + item + ".csv", newline = "") as csvfile:
			reader = csv.reader(csvfile, delimiter = ',')
			for row in reader:
				data[item].append(row)

	# data = data[500000 : ]

	data_rows_count = {}
	for item in DATA_CAT:
		data_rows_count[item] = len(data[item])
		print(">>> data rows count ({}):".format(item), data_rows_count[item])

	# Check one minute interval in doge
	prev_item = 0
	for item in data["doge"]:
		if int(item[0]) - prev_item > 60 * 1000 or prev_item == 0:
			print(">>>", int(item[0]) - prev_item)
			unixtime = int(item[0]) / 1000
			print(">>>", datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S'))
			# exit("Error time")
		prev_item = int(item[0])

	# Check one minute interval in btc
	prev_item = 0
	for item in data["btc"]:
		if int(item[0]) - prev_item > 60 * 1000 or prev_item == 0:
			print("ccc", int(item[0]) - prev_item)
			unixtime = int(item[0]) / 1000
			print(">>>", datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S'))
			# exit("Error time")
		prev_item = int(item[0])

	# Get min/max
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
		print(">>> min_close " + item, min_close[item])
		print(">>> max_close " + item, max_close[item])

	with open('min_max.csv', 'w') as f:
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

	# Cut data to dataframes for learning
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
				input_close_arr[item].append(input_close[item])
				output_close_arr[item].append(output_close[item])
				loop += SHIFT

	input_close_arr["doge"] = np.array(input_close_arr["doge"])
	input_close_arr["btc"] = np.array(input_close_arr["btc"])

	encode2 = np.vectorize(encode)
	input_close_arr["doge"] = encode2(input_close_arr["doge"], max_close["doge"])
	input_close_arr["btc"] = encode2(input_close_arr["btc"], max_close["btc"])

	# Add to DOGE BTC as next dimension
	X = np.array([input_close_arr["doge"][0], input_close_arr["btc"][0]])
	for index in range(1, len(input_close_arr["doge"])):
		print(index)
		X=np.append(X, [input_close_arr["doge"][index], input_close_arr["btc"][index]], axis = 0)
	X = np.reshape(X, (len(input_close_arr["doge"]),2 ,INPUT_LEN))

	y = np.array(output_close_arr["doge"], dtype = float)
	y = encode2(y, max_close["doge"])
	return X, y

def get_column(matrix, i):
	return [row[i] for row in matrix]

# Invert encoding
def decode(value, max):
	return value * max

# Encode data
def encode(value, max):
	result = float(value) / max
	return result

# Run evry epoch
def on_epoch_end(epoch, logs):
	print(">>>LOGS>>>", logs)

# Load trained weights
def model_load(model):
	if os.path.exists(FILEPATH):
		model.load_weights(FILEPATH)

# Train model
def model_train(model, X, y):
	checkpoint = ModelCheckpoint(FILEPATH, monitor = 'loss',
								 verbose = 1, save_best_only = True,
								 mode = 'min')

	print_callback = LambdaCallback(on_epoch_end = on_epoch_end)
	callbacks = [print_callback, checkpoint]
	model.fit(X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = callbacks)

model = lstm_hl(INPUT_LEN, OUTPUT_LEN)
model_load(model)
print(model.summary())
X, y = data_load()
model_train(model, X, y)


