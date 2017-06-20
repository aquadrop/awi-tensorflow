""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to run the model.

See readme.md for instruction on how to run the starter code.
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import inspect

import numpy as np
import tensorflow as tf

batch_size = 2

EOS = 6
VOL_SIZE = 7
EMBEDDING_SIZE = 10
ENCODER_SEQ_LENGTH = 2
ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
DECODER_SEQ_LENGTH = 1
DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

def random_one_hot_triple():
    zeros_a = np.zeros(10, dtype=np.float32)
    zeros_b = np.zeros(10, dtype=np.float32)
    zeros_c = np.zeros(10, dtype=np.float32)

    a = np.random.random_integers(5) - 1
    b = np.random.random_integers(5) - 1
    c = a + b

    print(a, b, c)
    zeros_a[a] = 1
    zeros_b[b] = 1
    zeros_c[c] = 1

    return zeros_a, zeros_b, zeros_c

def gen_triple():
    a = np.random.random_integers(5) - 1
    b = np.random.random_integers(5) - 1
    c = a + b
    return a, b, c

def plus_op_data():

    while True:
        encoder_inputs = []
        decoder_inputs = []
        for i in xrange(batch_size):
            a, b, c = gen_triple()
            encoder_inputs.append([a, b])
            decoder_inputs.append([c])

        yield np.asarray(encoder_inputs, dtype=np.int32).reshape(batch_size, 2), np.asarray(decoder_inputs, dtype=np.int32).reshape((batch_size, 1))

def single_cell(size=128):
    if 'reuse' in inspect.getargspec(
            tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)

def stacked_rnn(size=128):
    # return tf.contrib.rnn.MultiRNNCell([single_cell(size) for _  in range(2)])
    cells = list()
    cells.append(single_cell(size))
    cells.append(single_cell(size/2))
    return tf.contrib.rnn.MultiRNNCell(cells)

def one_out_stacked_rnn(size=128):
    cells = list()
    cells.append(single_cell(size))
    cells.append(single_cell(1))
    return tf.contrib.rnn.MultiRNNCell(cells)

def init_state(cell, batch_size):
    return cell.zero_state(batch_size=batch_size, dtype=tf.float32)

def train():

    embedding = tf.get_variable(
        "embedding", [VOL_SIZE, EMBEDDING_SIZE], dtype=tf.float32)

    encoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, 2), name="encoder_inputs")
    decoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, 1), name="decoder_inputs")

    encoder_embedding_vectors = tf.nn.embedding_lookup(embedding, encoder_inputs)
    decoder_embedding_vectors = tf.nn.embedding_lookup(embedding, decoder_inputs)

    encoder_cell = stacked_rnn(32)
    encoder_state = init_state(encoder_cell, batch_size)
    with tf.variable_scope("encoder") as scope:
        for time_step in xrange(ENCODER_NUM_STEPS):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            encoder_output, encoder_state = encoder_cell(encoder_embedding_vectors[:, time_step, :], encoder_state)
    #
    W2 = tf.Variable(np.random.rand(16, 1), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)

    logits_series = tf.matmul(encoder_output, W2) + b2  # Broadcasted addition
    #
    predictions = logits_series
    # #
    # loss = tf.reduce_mean(tf.square(tf.subtract(decoder_inputs, predictions)))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        # with tf.variable_scope('decoder') as scope:
        #     decoder_inputs = tf.placeholder(tf.float32, shape=[None, 1], name="decoder")
        #     decoder_cell = stacked_rnn(32)
        #     c = encoder_state
        #     decoder_output, decoder_state = decoder_cell(decoder_inputs, c)
        #
        #     W2 = tf.Variable(np.random.rand(16, 10), dtype=tf.float32)
        #     b2 = tf.Variable(np.zeros((1, 10)), dtype=tf.float32)
        #
        #     logits_series = tf.matmul(decoder_output, W2) + b2  # Broadcasted addition
        #     predictions_series = tf.nn.softmax(logits_series)
        #
        #     predictions = tf.cast(tf.argmax(predictions_series, axis=1), dtype=tf.float32)
        #
        #     loss = tf.reduce_mean(tf.square(tf.subtract(decoder_inputs, predictions)))
        #     train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        gen = plus_op_data()
        sess.run(tf.global_variables_initializer())
        i = 0
        for e_inputs, d_inputs in gen:
            # train_step.run(feed_dict={encoder_inputs: e_inputs, decoder_inputs: d_inputs})
            # if (i + 1) % 10 == 0:
            #     print(sess.run(loss, feed_dict={encoder_inputs: e_inputs, decoder_inputs:d_inputs}))
            #     save_path = saver.save(sess, "../model/model.ckpt")
            #     print("Model saved in file: %s" % save_path)
            # i = i + 1
            print(sess.run(encoder_output, feed_dict={encoder_inputs:e_inputs, decoder_inputs:d_inputs}))

def rnn_plus():
    return 0

def loss(decoder_inputs, decoder_outputs):
    diff = tf.metrics.mean_squared_error(decoder_inputs, decoder_outputs)
    return diff

if __name__ == "__main__":
    train()
    # a = np.random.rand(2,2)
    # x = tf.placeholder(tf.float32, shape=(2, 2))
    # y = tf.matmul(x, x)
    # with tf.Session() as sess:
    #     print(sess.run(y, feed_dict={x:a}))
