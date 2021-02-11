#!/usr/bin/env python3
from keras.callbacks import ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import json
import time
from numpy.lib.type_check import real
import requests

from numpy.lib.function_base import vectorize
from tensorflow.python.keras.backend import dtype
# Own library
from models.model import lstm_hl

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.system('python3 binance/download_doge_for_predict.py -x DOGEUSDT -s "24 hours 10 minutes ago UTC" -i 1m')

####################################################
# PARAMETERS
####################################################
INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 11
BATCH_SIZE = 128
FILEPATH = "weights_doge_btc.hdf5"
# List = name : column
DATA_CAT = {"doge" : 4, "btc" : 4}
####################################################

def sleep_anim(seconds):
    start = time.time()
    iteration = 0
    symbols = ['-', '\\', '|', '/']
    symbols = ['>    ', '>>   ', '>>>  ', '>>>> ', '>>>>>', '>>>> ', '>>>  ', '>>   ']
    go = True
    while go:
        now = time.time()
        print(symbols[iteration], str(seconds - int(now - start)).zfill(2), end="\r", flush=True)
        time.sleep(0.3)
        iteration += 1
        if iteration > len(symbols) - 1:
            iteration = 0
        if now - start > seconds:
            go = False

def load_min_max_doge ():
	# Read CSV file into array
	with open ("min_max_doge.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		row1 = next(reader)
		min_doge = row1[0]
		max_doge = row1[1]
	return min_doge, max_doge

# Load last data for prefict
def last_data_load (max_doge):
	data_doge_raw = []
	# Read CSV file into array
	with open ("doge_for_predict.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data_doge_raw.append (row)

	print (">>> rows all:", len(data_doge_raw))

	data_high_doge = get_column(data_doge_raw, 2)
	data_low_doge = get_column(data_doge_raw, 3)
	data_close_doge = get_column(data_doge_raw, 4)

	data_high_doge = data_high_doge[ : 1440]
	data_low_doge = data_low_doge[ : 1440]

	data_real = data_close_doge[1440 : ]

	data_high_doge_rows_count = len(data_high_doge)
	print (">>> data doge rows count:", data_high_doge_rows_count)
	print (">>> data real rows count:", len(data_real))

	input_high_doge = np.array(data_high_doge)
	input_low_doge = np.array(data_low_doge)

	encode2 = np.vectorize(encode)
	input_high_doge = encode2(data_high_doge, max_doge)
	input_low_doge = encode2(data_low_doge, max_doge)

	X = np.array(([input_high_doge], [input_low_doge]), dtype = float)

	X = np.reshape(X, (1, 2, INPUT_LEN))
	print("X", X)
	print("X.shape", X.shape)
	# exit("END")
	return X, data_real

def get_column(matrix, i):
	return [row[i] for row in matrix]

# Invert encoding
def decode (value, max):
	return value * max

# Encode data
def encode (value, max):
	result = float(value) / max
	return result

# Load trained weights
def model_load (model):
	if os.path.exists (FILEPATH):
		model.load_weights (FILEPATH)

def predict (X):
	# X = np.expand_dims (X, axis = 0)
	print(">>> X.shape", X.shape)
	result = model.predict (X, batch_size = BATCH_SIZE, verbose = 1)
	return result

min_doge, max_doge = load_min_max_doge()
min_doge, max_doge = float(min_doge), float(max_doge)

print("max_doge", max_doge)

model = lstm_hl (INPUT_LEN, OUTPUT_LEN)
model_load(model)

X, data_real = last_data_load(max_doge)

start_point = decode(X[0][0][1439], max_doge)
print(">>> start_point", start_point)
result = predict(X)
result_decoded = decode(result[0], max_doge)
result_decoded = np.insert(result_decoded, 0, start_point)
print("RESULT", result_decoded)
data_real = np.array(data_real, dtype = float)
data_real = np.insert(data_real, 0, start_point)
print("REAL", data_real)

plt.plot(data_real, c = 'green')
plt.plot(result_decoded, c = 'blue')
plt.show()
