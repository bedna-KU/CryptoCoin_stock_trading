#!/usr/bin/env python3
from keras.utils import plot_model
import os
# from tf.keras.utils import plot_model
# Own library
from models.model import lstm_medium

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
MAX_VALUE = 1
ACTIVATION = "relu"
EPOCHS = 20
BATCH_SIZE = 128
FILEPATH = "weights.hdf5"

####################################################

model = lstm_medium (INPUT_LEN, OUTPUT_LEN)
print (model.summary())
dot_img_file = 'keras_lstm_model.png'
plot_model(model, to_file = dot_img_file, show_shapes = True)
