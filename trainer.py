import numpy as np
import tensorflow as tf
from data_formatter import Data_Formatter
import time, os
from rnn_model import RNN

#hyperparameters
file_name = 'training.txt'
epochs = 50 #number of times training data is completely sent thru 
learning_rate = 0.002 
batch_size = 50 # batchsize for minibatch gradient descent
layer_size = 150  #number of neurons for each LSTM hidden layer
num_layers = 3 #number of LSTM layers
num_steps = 200  #number of time steps in RNN 

data = Data_Formatter(file_name, batch_size, num_steps)
num_classes = len(data.unique_notes)

rnn = RNN(batch_size,num_steps,num_classes,num_layers,layer_size,learning_rate)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

for epoch in range(epochs):
	data.reset()
	for batch in range(data.num_batches):
		stime = time.time()
		x,y = data.next_batch()
		input_feed = {rnn.x: x, rnn.targets: y}
		train_loss, state = sess.run([rnn.cost, rnn.opt], input_feed)
		etime = time.time()
		print('batch %s/%s, epoch %s/%s, loss = %s, time/batch = %s' %(batch+1, data.num_batches, epoch, epochs, train_loss, (etime-stime)))
	checkpoint_path = os.path.join('saved_models', 'model.cpkt')
	saver.save(sess, checkpoint_path, global_step=epoch)
	print('model saved in %s' %(checkpoint_path))



