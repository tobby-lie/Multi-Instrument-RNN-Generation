# -*- coding: utf-8 -*-
"""Music_Generator_MIDI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19lrZ1LdiEkkuLRJ719shbZr5NR8dUT7e
"""

# Commented out IPython magic to ensure Python compatibility.
""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import time
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/My\ Drive/Colab_Notebooks_Music_Gen
# %ls

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all unique pitch names sorted
    pitchnames = sorted(set(item for item in notes))
 
    # Get length of unique notes
    n_vocab = len(set(notes))

    # Get network input and normalized input
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    # Get model for predicting
    model = create_network(normalized_input, n_vocab)
    
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # define sequence length of 100
    sequence_length = 100
    # empty list for network input
    network_input = []
    # empty list for output
    output = []
    # from 0 to one element before len(notes - sequence_length)
    # in increments of 1
    for i in range(0, len(notes) - sequence_length, 1):
        # sequence of length 100 from i to one element before i + sequence_length
        # in notes list
        sequence_in = notes[i:i + sequence_length]
        # sequence out is a single note that is the one right after
        # sequence_in
        sequence_out = notes[i + sequence_length]
        # network_input is int represenation of notes and chords from
        # sequence_in
        network_input.append([note_to_int[char] for char in sequence_in])
        # int representation of note or chord from sequence_out
        output.append(note_to_int[sequence_out])

    # number of patterns eq length of network_input
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('weights-improvement-198-0.0094-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    # value from 0 to length of input - 1
    start = numpy.random.randint(0, len(network_input)-1)

    # create dict with key int and value note
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    # pattern is the randomly chosen starting point in the input
    pattern = network_input[start]
    # empty list for prediction output
    prediction_output = []

    # generate 500 notes
    # from 0 - 499
    for note_index in range(500):
        # input is the pattern chosen from some random starting point
        # reshaped for LSTM
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        # normalize input
        prediction_input = prediction_input / float(n_vocab)

        # form prediction from model
        prediction = model.predict(prediction_input, verbose=0)

        # get max value from prediction for index
        index = numpy.argmax(prediction)
        # the result is the note at index
        result = int_to_note[index]

        # append result to prediction_output list       
        prediction_output.append(result)

        # pattern appends index
        pattern.append(index)
        # pattern equals pattern from 1 to length of pattern
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5
    
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()