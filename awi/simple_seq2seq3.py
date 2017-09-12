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

This implementation learns NUMBER SORTING via seq2seq. Number range: 0,1,2,3,4,5,EOS

https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
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

batch_size = 64

EOS = 6
VOL_SIZE = 7
EMBEDDING_SIZE = 10
ENCODER_SEQ_LENGTH = 3
ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
DECODER_SEQ_LENGTH = ENCODER_SEQ_LENGTH + 1 ## plus 1 EOS
DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

HIDDEN_UNIT = 128
N_LAYER = 3

def one_hot_triple(index):
    zeros_a = np.zeros(VOL_SIZE, dtype=np.float32)
    zeros_a[index] = 1
    return zeros_a

def gen_triple():
    _input = np.random.random_integers(5, size=ENCODER_SEQ_LENGTH)
    _output = np.sort(_input)

    label = []
    for o in _output:
        label.append(one_hot_triple(o))
    label.append(one_hot_triple(EOS))

    _output = np.append(EOS, _output)
    return _input, _output, label

def plus_op_data():

    while True:
        encoder_inputs = []
        decoder_inputs = []
        labels = [] ## this is basically identical to decoder inputs tailed with EOS
        for i in xrange(batch_size):
            _input, _output, label= gen_triple()
            encoder_inputs.append(_input)
            decoder_inputs.append(_output)
            labels.append(label)

        yield np.asarray(encoder_inputs, dtype=np.int32).reshape(batch_size, ENCODER_SEQ_LENGTH), np.asarray(decoder_inputs, dtype=np.int32).reshape((batch_size, DECODER_SEQ_LENGTH)), np.asarray(labels)

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
    return tf.contrib.rnn.MultiRNNCell([single_cell(size) for _  in range(N_LAYER)])
    # cells = list()
    # cells.append(single_cell(size))
    # cells.append(single_cell(size/2))
    # return tf.contrib.rnn.MultiRNNCell(cells)

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

    labels_ = tf.placeholder(tf.float32, shape=(batch_size, DECODER_SEQ_LENGTH, VOL_SIZE))

    with tf.variable_scope("encoder") as scope:
        encoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, ENCODER_SEQ_LENGTH), name="encoder_inputs")
        encoder_embedding_vectors = tf.nn.embedding_lookup(embedding, encoder_inputs)
        encoder_cell = stacked_rnn(HIDDEN_UNIT)
        encoder_state = init_state(encoder_cell, batch_size)
        for time_step in xrange(ENCODER_NUM_STEPS):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            encoder_output, encoder_state = encoder_cell(encoder_embedding_vectors[:, time_step, :], encoder_state)

    last_encoder_state = encoder_state
    num_classes = VOL_SIZE
    loss = 0
    decoder_outputs = []
    with tf.variable_scope("decoder") as scope:
        decoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, DECODER_SEQ_LENGTH), name="decoder_inputs")
        decoder_embedding_vectors = tf.nn.embedding_lookup(embedding, decoder_inputs)
        decoder_cell = stacked_rnn(HIDDEN_UNIT)
        ## use softmax to map decode_output to number(0-5,EOS)
        W2 = tf.Variable(np.random.rand(HIDDEN_UNIT, num_classes), dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)
        for time_step in xrange(DECODER_NUM_STEPS):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            else:
                decoder_state = last_encoder_state
            decoder_output, decoder_state = decoder_cell(decoder_embedding_vectors[:, time_step, :], decoder_state)
            logits_series = tf.matmul(decoder_output, W2) + b2  # Broadcasted addition
            # logits_series = tf.nn.softmax(logits_series, dim=1)
            y_ = labels_[:, time_step, :]

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_series))
            loss = loss + cross_entropy

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        gen = plus_op_data()
        sess.run(tf.global_variables_initializer())
        i = 0
        for e_inputs, d_inputs, labels in gen:
            train_step.run(feed_dict={encoder_inputs: e_inputs, decoder_inputs: d_inputs, labels_: labels})
            if (i + 1) % 100 == 0:
                print(sess.run(loss, feed_dict={encoder_inputs: e_inputs, decoder_inputs: d_inputs, labels_: labels}))
                save_path = saver.save(sess, "../model/model")
            i = i + 1

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("../model"))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def run_sort():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        _check_restore_parameters(sess, saver)

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
