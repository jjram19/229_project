""" Code referenced from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py"""

from __future__ import print_function
import random, os, csv
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import plot, figure, show, title

# Parameters
learning_rate = 0.0025
num_steps = 250
BATCH_SIZE = 20
test_size = BATCH_SIZE*10
epoch_range = 20

# Network Parameters
cost_threshold = 100000
n_hidden_1 = 100
sample_factor = 1
num_input = 250000 # size of x input
num_classes = 2 # all genres


# Data formatting
y_data_total = np.genfromtxt('y_data.csv', delimiter = ',')
x_data_total = np.zeros((y_data_total.shape[0],num_input))
line_num = 0
with open('fft_data.csv', 'r') as _filehandler:
	csv_file_reader = csv.reader(_filehandler)
	for line, row in enumerate(csv_file_reader):
		x_data_total[line] = row

rng_state = np.random.get_state(); np.random.shuffle(x_data_total)
np.random.set_state(rng_state); np.random.shuffle(y_data_total)
x_test = x_data_total[:test_size]; x_data = x_data_total[test_size:]
y_test = y_data_total[:test_size]; y_data = y_data_total[test_size:]

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	return out_layer

def get_training_batch(batch_size):
	batch_range = random.sample(range(0, y_data.shape[0]), batch_size)
	x_train = x_data[batch_range]
	y_train = y_data[batch_range]
	return x_train, y_train

#construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

saver = tf.train.Saver()

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
epoch_cost = np.zeros(epoch_range)
# Start of training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	iteration = 0
	for epoch in range(epoch_range):
		cost = 0.0
		for step in range(1, num_steps+1):
			batch_x, batch_y = get_training_batch(BATCH_SIZE)
			_, c = sess.run([train_op, loss_op], feed_dict={X:batch_x, Y:batch_y})
			iteration += 1
			cost += c
		
		cost = cost/num_steps
		
		epoch_cost[epoch] = cost
		print("Total Epoch Cost: {:.9f}".format(cost))
		

	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Train Accuracy", accuracy.eval({X: x_train, Y: t_train}))
	print("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

	save_path = saver.save(sess, "./output/trained_binary.ckpt")

	sess.close()
	print("optimization finished")




