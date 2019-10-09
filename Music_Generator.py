import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding

data_directory = "/Users/tobbylie/Documents/CSCI_5931/Final_Project_DL"
data_file = "jigs.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = "/Users/tobbylie/Documents/CSCI_5931/Final_Project_DL/"
BATCH_SIZE = 16
SEQ_LENGTH = 64

def make_model(unique_chars):
	model = Sequential()

	model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (1, 1))) 

	model.add(LSTM(256, return_sequences = True, stateful = True))
	model.add(Dropout(0.2))

	model.add(LSTM(256, return_sequences = True, stateful = True))
	model.add(Dropout(0.2))

	model.add(LSTM(256, stateful = True)) 
	#remember, that here we haven't given return_sequences = True because here we will give only one character to generate the
	#sequence. In the end, we just have to get one output which is equivalent to getting output at the last time-stamp. So, here
	#in last layer there is no need of giving return sequences = True.
	model.add(Dropout(0.2))

	model.add((Dense(unique_chars)))
	model.add(Activation("softmax"))

	return model

def generate_sequence(epoch_num, initial_index, seq_length):
	with open(os.path.join(data_directory, charIndex_json)) as f:
		char_to_index = json.load(f)
	index_to_char = {i:ch for ch, i in char_to_index.items()}
	unique_chars = len(index_to_char)

	model = make_model(unique_chars)
	model.load_weights(model_weights_directory + "Weights_{}.h5".format(epoch_num))

	sequence_index = [initial_index]

	for _ in range(seq_length):
		batch = np.zeros((1, 1))
		batch[0, 0] = sequence_index[-1]

		predicted_probs = model.predict_on_batch(batch).ravel()
		sample = np.random.choice(range(unique_chars), size = 1, p = predicted_probs)

		sequence_index.append(sample[0])

	seq = ''.join(index_to_char[c] for c in sequence_index)

	cnt = 0
	for i in seq:
		cnt += 1
		if i == "\n":
			break
	seq1 = seq[cnt:]

	#above code is for ignoring the starting string of a generated sequence. This is because we are passing any arbitrary 
	#character to the model for generating music. Now, the model start generating sequence from that character itself which we 
	#have passed, so first few characters before "\n" contains meaningless word. Model start generating the music rhythm from
	#next line onwards. The correct sequence it start generating from next line onwards which we are considering.

	cnt = 0
	for i in seq1:
		cnt += 1
		if i == "\n" and seq1[cnt] == "\n":
			break
	seq2 = seq1[:cnt]

    #Now our data contains three newline characters after every tune. So, the model has leart that too. So, above code is used for
    #ignoring all the characters that model has generated after three new line characters. So, here we are considering only one
    #tune of music at a time and finally we are returning it..

	return seq2

ep = int(input("1. Which epoch number weight you want to load into the model(10, 20, 30, ..., 90). Small number will generate more errors in music: "))
ar = int(input("\n2. Enter any number between 0 to 86 which will be given as initial charcter to model for generating sequence: "))
ln = int(input("\n3. Enter the length of music sequence you want to generate. Typical number is between 300-600. Too small number will generate hardly generate any sequence: "))

music = generate_sequence(ep, ar, ln)

print("\nMUSIC SEQUENCE GENERATED: \n")

print(music)









