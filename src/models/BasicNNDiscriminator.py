import tensorflow as tf
import numpy as np 
import math
import pandas as pd

alpha = 0.01

def forward_propagate(inputs, W_1, b_1, W_2, b_2):
	z_1 = tf.matmul(W_1, inputs) + b_1
	a_1 = tf.nn.relu(z_1)
	z_2 = tf.matmul(W_2, a_1)+b_2
	return tf.sigmoid(z_2), z_2

def initialize_parameters(inputs_size = None, labels_size = None):
	W_1 = tf.get_variable(name="W1",shape=(100, inputs_size), initializer = tf.contrib.layers.xavier_initializer())
	b_1 = tf.get_variable(name="b1",shape=(100, 1), initializer = tf.zeros_initializer())
	W_2 = tf.get_variable(name="W2",shape=(1, 100), initializer = tf.contrib.layers.xavier_initializer())
	b_2 = tf.get_variable(name="b2",shape=(1, 1), initializer = tf.zeros_initializer())
	return W_1, b_1, W_2, b_2

def get_placeholders(inputs_size = None, inputs_batch = None, labels_size = None, labels_batch = None):
	inputs_placeholder = tf.placeholder(tf.float32,(inputs_size, None))
	labels_placeholder = tf.placeholder(tf.float32,(labels_size, None))
	return inputs_placeholder, labels_placeholder

def main(print_cost = True):

	traindata = np.asarray(pd.read_csv("trainData.csv", header = None))
	#Data inputs
	inputs_train, labels_train = traindata[:,1:-1], traindata[:,-1]
	labels_train = np.reshape(labels_train, (-1, 1))
	'''print(traindata.shape)
	print(traindata[11][0])
	print(inputs_train.shape)
	print(labels_train.shape)'''


	inputs_size = inputs_train.shape[1]
	inputs_batch = inputs_train.shape[0]

	labels_size = labels_train.shape[1]
	labels_batch = labels_train.shape[0]

	# define placeholders
	inputs, labels = get_placeholders(inputs_size, inputs_batch, labels_size, labels_batch)

	# define parameters 
	parameters = initialize_parameters(inputs_size, labels_size)

	# forward prop
	score, logit = forward_propagate(inputs, *parameters)
	pred = tf.round(score)

	# back prop
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,logits = logit)

	# update
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(10):
			epoch_cost = 0
			correct_preds = 0
			for j in range(len(inputs_train)):
				curr_input = np.reshape(inputs_train[j], (-1, 1))
				curr_label = np.reshape(labels_train[j], (-1, 1))
				prediction, _, example_cost = sess.run([pred, optimizer,loss], feed_dict={inputs: curr_input, labels: curr_label})
				if prediction == curr_label:
					correct_preds += 1
				epoch_cost += example_cost
				if j%10 == 0:
					print("Smoothed loss at Epoch " + str(i) + " and example " + str(j) + " : " + str(float(epoch_cost)/(j+1)))
			print("Smoothed loss at Epoch " + str(i) + " " + str(float(epoch_cost)/len(inputs_train)))
			print("Accuracy at Epoch " + str(i) + " " + str(100*float(correct_preds)/len(inputs_train)))

if __name__ == '__main__':
	main(True)