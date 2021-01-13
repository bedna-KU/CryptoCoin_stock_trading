#!/usr/bin/env python3
import os
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import time
import requests
import json
from keras.callbacks import ModelCheckpoint, LambdaCallback
from numpy.lib.function_base import vectorize

# Own library
from models.model import lstm_medium

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.system('python3 binance/download_data_for_predict.py -x DOGEUSDT -s "24 hours ago UTC" -i 1m')

####################################################
# PARAMETERS
####################################################

INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 20
BATCH_SIZE = 128
FILEPATH = "weights.hdf5"

####################################################

def sleep_anim(seconds):
    start = time.time()
    iteration = 0
    symbols = ['-', '\\', '|', '/']
    symbols = ['>    ', '>>   ', '>>>  ', '>>>> ', '>>>>>', '>>>> ', '>>>  ', '>>   ']
    go = True
    while go:
        now = time.time()
        print(symbols[iteration], seconds - int(now - start), end="\r", flush=True)
        time.sleep(0.3)
        iteration += 1
        if iteration > len(symbols) - 1:
            iteration = 0
        if now - start > seconds:
            go = False

# Load data
def data_load ():
	print(">>>>>>> data_load")
	data = []
	# Read CSV file into array
	with open ("file_new.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data.append (row)

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

	return min_close, max_close

# Load last data for prefict
def last_data_load (min_close, max_close):
	print(">>>>>>> last_data_load")
	data = []
	# Read CSV file into array
	with open ("file_for_predict.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data.append (row)

	data_rows_count = len(data)
	print (">>> data rows count:", data_rows_count)

	data_close = get_column(data, 4)
	input_close_arr = data_close

	print(">>> input_close_arr", len(input_close_arr))

	# To ndarray
	X = np.array(input_close_arr)
	encode2 = np.vectorize(encode)
	input_close_arr = encode2(X, max_close)
	return input_close_arr

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
	result = float(value) / max
	return result

# Run evry epoch
def on_epoch_end (epoch, logs):
	print (">>>LOGS>>>", logs)

# Load trained weights
def model_load (model):
	if os.path.exists (FILEPATH):
		model.load_weights (FILEPATH)

def predict (X):
	print(">>> X", X)
	print(">>> X.shape", X.shape)
	X = np.expand_dims (X, axis = 0)
	X = np.expand_dims (X, axis = 2)
	print(">>> X.shape", X.shape)
	result = model.predict (X, batch_size = BATCH_SIZE, verbose = 1)
	return result

model = lstm_medium (INPUT_LEN, OUTPUT_LEN)
model_load(model)
min_close, max_close = data_load()
X = last_data_load(min_close, max_close)

start_point = decode(X[1339], max_close)

print("***", X.shape)
result = predict(X)
result_decoded = decode(result[0], max_close)
result_decoded = np.insert(result_decoded, 0, start_point)
print("RESULT", result_decoded)

real = start_point
print("### start_point", start_point)

sleep_anim(60)

for n in range(0, 10):
	print(">>> LOOP", n)
	r = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=DOGEUSDT')
	json_data = json.loads(r.text)
	price = json_data["price"]
	print (price)
	print("RESULT", result_decoded)
	real = np.append(real, float(price))
	print("REAL", real)
	sleep_anim(60)

plt.plot(result_decoded, c = 'blue')
plt.plot(real, c = 'green')
plt.text(2,4,'This text starts at point (2,4)')
# plt.draw()
# plt.pause(0.001)
plt.show()
# plt.close()
