import numpy as np
import tensorflow as tf
from data_formatter import Data_Formatter
import time, os
from rnn_model import RNN

#hyperparameters
file_name = 'testtraining.txt'
epochs = 20 #number of times training data is completely sent thru 
learning_rate = 0.002 
batch_size = 200 # batchsize for minibatch gradient descent
layer_size = 128  #number of neurons for each LSTM hidden layer
num_layers = 2 #number of LSTM layers
num_steps = 50  #number of time steps in RNN 

num_notes = 100
music_book_name = 'collection1.abc'

data = Data_Formatter(file_name, batch_size, num_steps)
num_classes = len(data.unique_notes)

rnn = RNN(batch_size,num_steps,num_classes,num_layers,layer_size,learning_rate, training=False)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

checkpoint= tf.train.get_checkpoint_state('saved_models')
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, checkpoint.model_checkpoint_path)
print('restored session!')

save_dir = os.path.join('songs', music_book_name)

file = open(save_dir, 'w')
file.write(rnn.generate_notes(sess,num_notes,data.notes_to_ind, data.ind_to_notes))
file.close()
