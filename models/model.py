import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, LSTM, RepeatVector, TimeDistributed, Input, Convolution2D, MaxPooling2D, InputLayer
from keras.optimizers import Adam

def cnn_3 (max_len, max_value, activation):
	model = Sequential ()
	model.add (Conv2D (32, kernel_size = (3, 3), padding = "valid", strides = (1, 1), activation = activation, input_shape = (max_len, max_value + 1, 1)))
	model.add (Conv2D (64, kernel_size = (3, 3), padding = "valid", strides = (1, 1), activation = activation))
	model.add (Conv2D (128, kernel_size = (3, 3), padding = "valid", strides = (1, 1), activation = activation))
	model.add (Flatten ())
	model.add (Dense (1000, activation = activation))
	model.add (Dense (max_len, activation = activation))

	model.compile (optimizer = Adam (),
				   loss = 'mse',
				   metrics = ['accuracy'])
	return model

def lstm_easy (input_len, output_len):
	# Configure the neural network model
	model = Sequential()

	model.add(LSTM(100, return_sequences = True, input_shape = (input_len, 1)))
	model.add(LSTM(100, return_sequences = False))
	model.add(Dense(25, activation = 'relu'))
	model.add(Dense(1))

	# Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

def lstm_medium (input_len, output_len):
	# Initialising the RNN
	model = Sequential()# Adding the first LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True, input_shape = (input_len, 1)))

	# Adding a second LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True))
	model.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True))
	model.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50))
	model.add(Dropout(0.2))

	# Adding the output layer
	model.add(Dense(units = output_len))
	# Compiling the RNN
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	return model

def lstm_hl (input_len, output_len):
	print("XXX input_len", input_len)
	# Initialising the RNN
	model = Sequential()# Adding the first LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True, input_shape = (2, input_len)))

	# Adding a second LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True))
	model.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50, return_sequences = True))
	model.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularisation
	model.add(LSTM(units = 50))
	model.add(Dropout(0.2))

	# Adding the output layer
	model.add(Dense(units = output_len))
	# Compiling the RNN
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	return model

def cnn_lstm (input_len, output_len):
	model = Sequential()
	model.add(InputLayer(input_shape=(input_len, 1)))
	model.add(TimeDistributed(Convolution2D(64, (3, 3))))
	model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
	model.add(LSTM(10))
	model.add(Dense(3))