# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:16:36 2017

@author: solanki
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 5
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


a = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')
def recurrent_neural_network(a):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    a = tf.transpose(a, [1,0,2])
    a = tf.reshape(a, [-1, chunk_size])
    a = tf.split(a, n_chunks, 0)

    self.lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(self.num_cells_1)
    outputs, states = rnn.static_rnn(lstm_cell, a, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output
def train_neural_network(a):
    prediction = recurrent_neural_network(a)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_a, epoch_b = mnist.train.next_batch(batch_size)
                epoch_a = epoch_a.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={a: epoch_a, b: epoch_b})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(b, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(a)

