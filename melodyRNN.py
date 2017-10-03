import numpy as np
import random
import collections
import time


#generate dictionaries to represent characters numerically and vice versa
def build_dataset(file):
	data = open(file, 'r').read()
	unique_notes = list(set(data))
	dictionary = dict()
	for i in range(len(unique_notes)):
		dictionary[unique_notes[i]] = i
	rev_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary,rev_dictionary
