#!/usr/bin/env python3
from keras.callbacks import ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import json
import time
import requests

from numpy.lib.function_base import vectorize
# Own library
from models.model import lstm_hl

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.system('python3 binance/download_doge_for_predict.py -x DOGEUSDT -s "24 hours ago UTC" -i 1m')
os.system('python3 binance/download_btc_for_predict.py -x BTCUSDT -s "24 hours ago UTC" -i 1m')

####################################################
# PARAMETERS
####################################################
INPUT_LEN = 1440
OUTPUT_LEN = 10
SHIFT = 10
EPOCHS = 11
BATCH_SIZE = 128
FILEPATH = "weights_hl.hdf5"# List = name : column
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
        print(symbols[iteration], seconds - int(now - start), end="\r", flush=True)
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
		row2 = next(reader)
		min_doge = row1[0]
		max_doge = row1[1]
		min_btc = row2[0]
		max_btc = row2[1]
	return min_doge, max_doge, min_btc, max_btc

# Load last data for prefict
def last_data_load (max_doge, max_btc):
	print(">>>>>>> last_data_load")
	data_doge = []
	# Read CSV file into array
	with open ("doge_for_predict.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data_doge.append (row)

	data_btc = []
	with open ("btc_for_predict.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data_btc.append (row)

	data_doge_rows_count = len(data_doge)
	data_btc_rows_count = len(data_btc)
	print (">>> data doge rows count:", data_doge_rows_count)
	print (">>> data btc rows count:", data_btc_rows_count)

	doge_close = get_column(data_doge, 4)
	btc_close = get_column(data_btc, 4)

	input_doge_arr = np.array(doge_close)
	input_btc_arr = np.array(btc_close)

	# input_doge_arr = np.reshape(input_doge_arr, (1440, 1))
	# input_doge_arr = np.reshape(input_doge_arr, (1, 1440))
	# input_btc_arr = np.reshape(input_btc_arr, (1440, 1))
	# input_btc_arr = np.reshape(input_btc_arr, (1, 1440))
	print("shape", input_btc_arr.shape)

	encode2 = np.vectorize(encode)
	input_doge_arr_enc = encode2(input_doge_arr, max_doge)
	input_btc_arr_enc = encode2(input_btc_arr, max_btc)

	# print("aaa", input_doge_arr)

	X = np.array(([input_doge_arr_enc], [input_btc_arr_enc]), dtype = float)
	# X = np.concatenate((input_doge_arr_enc, input_btc_arr_enc), axis=1)

	# X = np.reshape(X, (INPUT_LEN, 2))
	print("eee", X)
	exit()

	X = np.reshape(X, (1, INPUT_LEN, 2))
	print("eee", X)
	print("eee", X.shape)
	# exit()
	return X

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

min_doge, max_doge, min_btc, max_btc = load_min_max_doge()
min_doge, max_doge, min_btc, max_btc = float(min_doge), float(max_doge), float(min_btc), float(max_btc)

print("max_doge", max_doge)
print("max_btc", max_btc)

model = lstm_hl (INPUT_LEN, OUTPUT_LEN)
model_load(model)

X = last_data_load(max_doge, max_btc)

# print("XXX", decode(X[0][1440][0], max_doge))
print("XXX shape", X.shape)
# print("XXX", decode(X[1339][0], max_doge))
# print("X[1339][0]", X[1339][1], decode(X[1339][0], max_doge))
start_point = decode(X[0][1339], max_doge)

# exit()

print("***", X.shape)
result = predict(X)
result_decoded = decode(result[0], max_doge)
result_decoded = np.insert(result_decoded, 0, start_point)
print("RESULT", result_decoded)

real = start_point
print("### start_point", start_point)

sleep_anim(60)

for n in range(10):
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
plt.show()
