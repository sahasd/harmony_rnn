# Imports
import numpy as np
from numpy.random import choice
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os

class RNN():
	def __init__(self,batch_size,time_steps,num_classes,num_layers,layer_size,alpha, training=True):

		self.batch_size = batch_size
		self.time_steps = time_steps
		self.num_classes = num_classes
		self.num_layers = num_layers
		self.layer_size = layer_size
		self.alpha = alpha

		if not training:
			self.batch_size = 1
			self.time_steps = 1 

		self.x = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='input')
		self.targets = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='input')

		W = tf.get_variable('Weight', [self.layer_size, self.num_classes])
		b = tf.get_variable('Bias', [num_classes], initializer=tf.constant_initializer(0.0))

		embeddings = tf.get_variable('embedding_matrix', [self.num_classes, self.layer_size])
		rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x)

		self.cell = tf.nn.rnn_cell.LSTMCell(self.layer_size, state_is_tuple=True)
		self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=0.97, output_keep_prob=0.97) #dropout to reduce overfitting
		self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.num_layers, state_is_tuple=True)

		self.initial = self.cell.zero_state(self.batch_size, tf.float32)
		self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, initial_state=self.initial)

		self.rnn_outputs = tf.reshape(tf.concat(self.rnn_outputs, 1), [-1, layer_size])
		transposed_targets = tf.reshape(self.targets, [-1])

		self.logits = tf.matmul(self.rnn_outputs, W) + b
		self.distribution = tf.nn.softmax(self.logits)

		self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.rnn_outputs, labels=transposed_targets))
		self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.cost)

		# gradient clipping to prevent exploding gradient issue
		'''gradient_vectors = self.optimizer.compute_gradients(self.cost)
		clipped_gradients = [(self.ClipIfNotNone(grad), var) for grad, var in gradient_vectors]
		self.opt = self.optimizer.apply_gradients(clipped_gradients)'''


	def generate_notes(self,sess, num_notes, notes_to_ind, ind_to_notes):
	
		state = sess.run(self.cell.zero_state(1, tf.float32))

		note = 'T'
		music_sheet = note

		for n in range(num_notes):
			x = np.zeros((1,1))
			x[0,0] = notes_to_ind[note]
			input_feed = {self.x: x, self.initial: state}
			prob_dist, state = sess.run([self.distribution, self.final_state], input_feed)
			note = ind_to_notes[self.weighted_random(prob_dist[0])]
			music_sheet += note
			print(note)

		return music_sheet

	def weighted_random(self,arr):  #randomly select a character based on how likely it is come next
		t = np.cumsum(arr)
		s = np.sum(arr)
		return(int(np.searchsorted(t, np.random.rand(1)*s)))

	def ClipIfNotNone(self,grad):
		if grad is None:
			return grad
		return tf.clip_by_value(grad, -5, 5)
