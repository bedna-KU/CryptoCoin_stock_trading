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
	return row1[0], row1[1]

# Load last data for prefict
def last_data_load (max):
	print(">>>>>>> last_data_load")
	data = []
	# Read CSV file into array
	with open ("file_for_predict.csv", newline="") as csvfile:
		reader = csv.reader (csvfile, delimiter=',')
		for row in reader:
			data.append (row)

	data_rows_count = len(data)
	print (">>> data rows count:", data_rows_count)

	data_high = get_column(data, 2)
	data_low = get_column(data, 3)

	input_high_arr = np.array(data_high)
	input_low_arr = np.array(data_low)

	X = np.array([input_high_arr, input_low_arr], dtype = float)

	print("xxx", X.shape)

	X = np.reshape(X, (INPUT_LEN, 2))

	encode2 = np.vectorize(encode)
	X = encode2(X, max)

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

# Run evry epoch
def on_epoch_end (epoch, logs):
	print (">>>LOGS>>>", logs)

# Load trained weights
def model_load (model):
	if os.path.exists (FILEPATH):
		model.load_weights (FILEPATH)

def predict (X):
	X = np.expand_dims (X, axis = 0)
	print(">>> X.shape", X.shape)
	result = model.predict (X, batch_size = BATCH_SIZE, verbose = 1)
	return result

min, max = load_min_max_doge()
min, max = float(min), float(max)

model = lstm_hl (INPUT_LEN, OUTPUT_LEN)
model_load(model)

X = last_data_load(max)

start_point = decode(X[1339][0], max)

print("***", X.shape)
result = predict(X)
result_decoded = decode(result[0], max)
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
