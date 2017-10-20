# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import Data_Utils
import time, os

class RNN():
	def __init__(self,epochs,batch_size,time_steps,num_layers,layer_size,alpha, load_model=False):
		self.epochs = epochs
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.num_layers = num_layers
		self.layer_size = layer_size
		self.alpha = alpha

	if load_model is False:
		self.x = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='input')
		self.targets = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='input')

		W = tf.get_variable('Weight', [layer_size, num_classes])
		b = tf.get_variable('Bias', [num_classes], initializer=tf.constant_initializer(0.0))

		embeddings = tf.get_variable('embedding_matrix', [self.num_classes, self.layer_size])
		rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x)

		cell = tf.nn.rnn_cell.LSTMCell(self.layer_size, state_is_tuple=True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

		initial = cell.zero_state(self.batch_size, tf.float32)
		rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=initial)

		rnn_outputs = tf.reshape(tf.concat(rnn_outputs, 1), [-1, args.rnn_size])

        self.logits = tf.matmul(rnn_outputs, W) + b

		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_outputs, labels=self.targets))
		optimizer = tf.train.AdamOptimizer(self.alpha).minimize(cost)








