import tensorflow
from tensorflow.contrib import rnn
import numpy as np
import random

#training paramaters
learning_rate = 0.001
epochs = 10000
display_step = 200

#generate dictionaries to represent characters numerically and vice versa
def build_dataset(file):
	data = open(file, 'r').read()
	unique_notes = list(set(data))
	dictionary = dict()
	for i in range(len(unique_notes)):
		dictionary[unique_notes[i]] = i
	rev_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary,rev_dictionary

dictionary, rev_dictionary = build_dataset('sample_training.txt')
num_notes = len(dictionary)

#hyperparamaters
num_input = num_notes # 1 hot vector representing music note
timesteps = num_notes
hidden_cells = 128
num_classes = num_notes 

X = tf.placeholder('float', [None, timesteps, num_input])
Y = tf.placeholder('float', [None, num_classes]) 



