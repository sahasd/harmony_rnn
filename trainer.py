# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import Data_Utils
import time, os

#hyperparameters
file_name = 'training.txt'
epochs = 2 #number of times training data is completely sent thru 
learning_rate = 1e-3  #alpha = 0.0001
batch_size = 32 # batchsize for minibatch gradient descent
layer_size = 100  #number of neurons for each LSTM hidden layer
num_layers = 3 #number of LSTM layers
num_steps = 200  #number of time steps in RNN 

dat_util = Data_Utils(file_name, batch_size, num_steps)
num_classes = len(dat_util.unique_notes)

#CORE RNN MODEL DEFINED HERE
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input')
y_ = tf.placeholder(tf.int32, [batch_size, num_steps], name='correct_labels')

W = tf.get_variable('Weight', [layer_size, num_classes])
b = tf.get_variable('Bias', [num_classes], initializer=tf.constant_initializer(0.0))

#convert to matrix of 1 hot vectors
embeddings = tf.get_variable('embedding_matrix', [num_classes, layer_size])
rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

#forward pass+dropout
cell = tf.nn.rnn_cell.LSTMCell(layer_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob= tf.constant(1.0))

initial = cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=initial)

#reshape for simpler math in backprop
rnn_outputs = tf.reshape(rnn_outputs, [-1, layer_size])
y_transposed = tf.reshape(y_, [-1])
y = tf.nn.softmax(tf.matmul(rnn_outputs, W) + b) #predicted labels

#backprop
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_transposed))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for e in range(epochs):
	for n in range(dat_util.num_batches):
		print(n)
		x_train,y_train = dat_util.next_batch()
		print(len(x_train[-1]))
		predicted,actual = sess.run([y,y_], feed_dict={x:x_train, y_:y_train})
		print(predicted[0])
		print(len(predicted), len(actual))
	dat_util.reset()