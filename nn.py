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
test_size = BATCH_SIZE*5
display_step = 20
epoch_range = 30

# Network Parameters
n_hidden_1 = 100
n_hidden_2 = 100
# n_hidden_3 = 100
# sample_factor = 100
# num_input = 250000/sample_factor # size of x input
num_input = 250000
num_classes = 2 # all genres



# rap_range = range(1872, 2397)
total_songs = 7143 - len(vals_ignored)
# Data formatting
y_data_temp = np.genfromtxt('avg_y_data.csv', delimiter = ',')
x_data_total = np.zeros((total_songs,num_input))
# y_data_total = np.zeros((total_songs, num_classes))
# line_num = 0
with open('avg_fft_data.csv', 'r') as _filehandler:
	csv_file_reader = csv.reader(_filehandler)
	for line, row in enumerate(csv_file_reader):
		if(line in vals_ignored): continue
		if(line in rap_range):
			y_data_total[line_num][1] = 1
		else:
			y_data_total[line_num][0] = 1
		x_data_total[line_num] = row
		# x_data_total[line] = row
		line_num += 1
		if line % 100 == 0: print("row: " + str(line) + " added.")
		# print("Line added: " + str(line_num) + " Line number: " + str(line))

print(x_data_total.shape)
print(y_data_total.shape)
rng_state = np.random.get_state(); np.random.shuffle(x_data_total)
np.random.set_state(rng_state); np.random.shuffle(y_data_total)
x_test = x_data_total[:test_size]; x_data = x_data_total[test_size:]
y_test = y_data_total[:test_size]; y_data = y_data_total[test_size:]

# x_test = np.reshape(x_test, [test_size, num_input/sample_factor])
# y_test = np.reshape(y_test, [test_size, num_classes])


X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

def get_training_batch(batch_size):
	batch_range = random.sample(range(0, y_data.shape[0]), batch_size)
	x_train = x_data[batch_range]
	y_train = y_data[batch_range]

	# x_train = np.reshape(x_train, [batch_size, num_input])
	# y_train = np.reshape(y_train, [batch_size, num_classes])
	return x_train, y_train

#construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels=Y))
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
		if(cost <= 50000): break
		epoch_cost[epoch] = cost
		print("Total Epoch Cost: {:.9f}".format(cost))
		

	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

	sess.close()
	print("optimization finished")



