import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

import time
import numpy
import sys

numpy.set_printoptions(threshold=sys.maxsize)

data_directory = "/Users/tobbylie/Documents/CSCI_5931/Final_Project_DL"
data_file = "jigs.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = "/Users/tobbylie/Documents/CSCI_5931/Final_Project_DL"
# batch size and seq length can be anything we define them to be
BATCH_SIZE = 16
SEQ_LENGTH = 64

#-------------------------------------------------------------------------------------------
# Method: read_batches
#
#	Description:
#		From a given array of all characters and number of unique characters, this
#		method produces batches for model to train in batches. Each batch is of size 16
#		with X being 16 batches each containing 64 length sequences of characters based
#		on the data, Y being 16 batches each containing 64 length sequences, also containing
#		a third dimension representing a character from 87 unique characters that is the 
#		correct next character in the sequence, this is represented in one hot encoding
#		format.
#
#	parameters:
#		all_chars - numpy array of all characters form data file
#		unique_chars - variable representing number of unique characters in data file
#
#	returns (yields in order to return value and continue execution):
#		X - batch of quantity 16, each batch contains a sequence of length 64
#		Y - for each sequence of length 64 in each batch of 16 batches, each
#			sequence character in each batch must have a correct label which 
#			is the next character in the sequence, this will be one hot encoded
#-------------------------------------------------------------------------------------------
def read_batches(all_chars, unique_chars):
	# length equals all characters in data 
	length = all_chars.shape[0]
	# number of character in batch is equal to length/BATCH_SIZE
	# for example 155222/16 = 9701
	batch_chars = int(length / BATCH_SIZE)

	# batch_chars - SEQ_LENGTH = 9701 - 64 = 9637
	# (0, 9637, 64) from 0 to 9637 in intervals of 64
	# number of batches = 151 => 9637/64
	for start in range(0, batch_chars - SEQ_LENGTH, 64):
		# (16,64) => with all zeros
		X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
		# (16, 64, 87) => with all zeros
		Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, unique_chars))
		# for each row in a batch since first dimension of X and Y are both 16
		for batch_index in range(0, 16):
			# each column in a batch => represents each character in that sequence
			# there are 64 characters in a sequence and each character must be defined
			for i in range(0, 64):
				# time-step character in a sequence
				# X at batch_index, i means X at a certain batch at character i
				# this is equal to, from all characters, we are taking which ever
				# batch we are at multiplied by batch_chars plus start plus i 
				# this represents taking size 9701 steps in 16 intervals offset by
				# start and i
				X[batch_index, i] = all_chars[batch_index * batch_chars + start + i]
				print(X[batch_index, i])
				# '1' because the correct label will be the next character in the sequence
				# so next character denoted by all_chars[batch_index * batch_chars + start + i + 1]
				# this is at batch index, at character i and at next character
				Y[batch_index, i, all_chars[batch_index * batch_chars + start + i + 1]] = 1
		# Suspends function's execution and sends a value to caller, but retains
		# enough state to enable function to resume where it left off
		yield X, Y

def built_model(batch_size, seq_length, unique_chars):
	model = Sequential()

	model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (batch_size,
		seq_length)))

	model.add(LSTM(256, return_sequences = True, stateful = True))
	model.add(Dropout(0.2))

	model.add(LSTM(256, return_sequences = True, stateful = True))
	model.add(Dropout(0.2))

	model.add(LSTM(256, return_sequences = True, stateful = True))
	model.add(Dropout(0.2))

	model.add(TimeDistributed(Dense(unique_chars)))
	model.add(Activation("softmax"))

	return model

def training_model(data, epochs = 80):
	# mapping character to index via a dictionary
	# char as key and index as value
	# set(data) produces an unordered collection of characters from data with no duplicates
	# that is then turned into a list and then sorted to be looped through
	char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
	# print out total number of unique characters in data
	print("Number of unique characters in our whole tunes database = {}".format(len(char_to_index))) #87

	# define a path to our charIndex_json file and put the contents of char_to_index into it
	with open(os.path.join(data_directory, charIndex_json), mode = "w") as f:
		json.dump(char_to_index, f)

	# create dict flipping keys and values from char_to_index so the keys
	# become values and values become keys
	index_to_char = {i: ch for (ch, i) in char_to_index.items()}
	# unique characeter is the number of elements in char_to_index
	unique_chars = len(char_to_index)

	# use method built_model to return an RNN model of specified batch size,
	# sequence length and with input dimension = unique_chars
	model = built_model(BATCH_SIZE, SEQ_LENGTH, unique_chars)
	# print summary of made model
	model.summary()
	# train the model 
	model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

	# go character by character in data and get index of that character as array element
	# assign this numpy array to all_characters
	all_characters = np.asarray([char_to_index[c] for c in data], dtype = np.int32)

	# total number of characters should be 155222
	print("Total number of characters = "+str(all_characters.shape[0]))

	# create three empty lists for epoch number, loss, and accuracy
	epoch_number, loss, accuracy = [], [], []

	# for each epoch print which epoch we are in 
	# and add epoch number to list of epoch numbers
	for epoch in range(epochs):
		print("Epoch {}/{}".format(epoch+1, epochs))
		# initialize final epoch loss and final epoch accuracy to 0 to accumulate 
		# for each epoch
		final_epoch_loss, final_epoch_accuracy = 0, 0
		epoch_number.append(epoch+1)

		# for i => index and tuple (x, y) => training batch
		for i, (x, y) in enumerate(read_batches(all_characters, unique_chars)):
			# update final epoch loss and final epoch accuracy from model trained from batch (x, y)
			#check documentation of train_on_batch here: https://keras.io/models/sequential/
			final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y)
			# print out batch, loos, accuracy retreived from train on batch
			# here, we are reading the batches one-by-one and training our model on each batch one-by-one.
			print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
		# add final epoch loss and final epoch accuracy to our loss and accuracy lists respectively
		loss.append(final_epoch_loss)
		accuracy.append(final_epoch_accuracy)

		#saving weights after every 10 epochs
		if (epoch + 1) % 10 == 0:
			# if directory does not exist then make it
			if not os.path.exists(model_weights_directory):
				os.makedirs(model_weights_directory)
				# save weights to .h5 file
			model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
			# specify which multple of 10 epoch
			print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))

	#creating dataframe and record all the losses and accuracies at each epoch
	log_frame = pd.DataFrame(columns = ["Epoch", "Loss", "Accuracy"])
	log_frame["Epoch"] = epoch_number
	log_frame["Loss"] = loss
	log_frame["Accuracy"] = accuracy
	log_frame.to_csv("/Users/tobbylie/Documents/CSCI_5931/Final_Project_DL/log.csv", index = False)


file = open(os.path.join(data_directory, data_file), mode = 'r')
data = file.read()
file.close()
if __name__ == "__main__":
	training_model(data)

log = pd.read_csv(os.path.join(data_directory, "log.csv"))






