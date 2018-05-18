import tensorflow as tf
import numpy as np 
import math

alpha = 0.01

def forward_propagate(inputs, W_1, b_1, W_2, b_2):
	z_1 = tf.matmul(W_1, inputs) + b_1
	a_1 = tf.nn.relu(z_1)
	z_2 = tf.matmul(W_2, a_1)+b_2
	return tf.sigmoid(z_2), z_2

def initialize_parameters(inputs_size = None, labels_size = None):
	W_1 = tf.get_variable(name="W1",shape=(100, inputs_size), initializer = tf.contrib.layers.xavier_initializer())
	b_1 = tf.get_variable(name="W1",shape=(100, 1), initializer = tf.zeros_initializer())
	W_2 = tf.get_variable(name="W1",shape=(1, 100), initializer = tf.contrib.layers.xavier_initializer())
	b_2 = tf.get_variable(name="W1",shape=(1, 1), initializer = tf.zeros_initializer())
	return W_1, b_1, W_2, b_2

def get_placeholders(inputs_size = None, inputs_batch = None, labels_size = None, labels_batch = None):
	inputs_placeholder = tf.get_placeholder(tf.float32,(inputs_size, inputs_batch))
	labels_placeholder = tf.get_placeholder(tf.float32,(labels_size, labels_batch))
	return inputs_placeholder, labels_placeholder

def main(print_cost = True):
	#Data inputs
	inputs_train, labels_train = #woody gives this to me


	inputs_size = inputs[1]
	inputs_batch = inputs[2]

	labels_size = inputs[1]
	labels_batch = inputs[2]

	# define placeholders
	inputs, labels = get_placeholders(inputs_size, inputs_batch, labels_size, labels)

	# define parameters 
	parameters = initialize_parameters(inputs_size, labels_size)

	# forward prop
	score, logit = forward_propagate(inputs, *parameters)
	pred = np.round_(score)

	# back prop
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,logits = logit)

	# update
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)

	init = tf.global_variables_initializer()

	with tf.session() as sess:
		sess.run(init)
		for i in range(100):
			epoch_cost = 0
			correct_preds = 0
			for j in range(len(inputs_train)):
				curr_input = inputs_train[j]
				curr_label = labels_batch[j]
				pred, _, example_cost = sess.run([pred, optimizer,loss], feed_dict={inputs: curr_input, labels: curr_label})
				if pred == curr_label:
					correct_preds += 1
				epoch_cost += example_cost
				if j%10 == 0:
					"Smoothed loss at Epoch " + str(i) + " and example " + str(j) + " : " + str(float(epoch_cost)/(j+1))
            "Smoothed loss at Epoch " + str(i) + str(float(epoch_cost)/len(inputs_train))
            "Accuracy at Epoch " + str(i) + str(float(correct_preds)/len(inputs_train))

if __name__ == '__main__':
	main(True)