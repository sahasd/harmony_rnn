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

num_notes = 1000
music_book_name = 'first_attempt.txt'

data = Data_Formatter(file_name, batch_size, num_steps)
num_classes = len(data.unique_notes)

rnn = RNN(batch_size,num_steps,num_classes,num_layers,layer_size,learning_rate, training=False)

checkpoint_path = os.path.join('saved_models', 'model.cpkt')
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, checkpoint_path)
print('restored session!')

save_dir = os.path.join('songs', music_book_name)

file = open(save_dir, 'w')
file.write(rnn.generate_notes(sess,100,data.notes_to_ind, data.ind_to_notes))
file.close()


