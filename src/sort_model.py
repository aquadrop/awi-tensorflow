""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to build the model

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import inspect

class SortModel:

    batch_size = 1

    EOS = 6
    VOL_SIZE = 7
    EMBEDDING_SIZE = 10
    ENCODER_SEQ_LENGTH = 3
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH = ENCODER_SEQ_LENGTH + 1  ## plus 1 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

    HIDDEN_UNIT = 128
    N_LAYER = 3

    TRAINABLE = True


    def __init__(self, trainable = True):
        print('initilizing model...')
        self.TRAINABLE = trainable

    def single_cell(self, size=128):
        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)

    def stacked_rnn(self, size=128):
        return tf.contrib.rnn.MultiRNNCell([self.single_cell(size) for _ in range(self.N_LAYER)])
        # cells = list()
        # cells.append(single_cell(size))
        # cells.append(single_cell(size/2))
        # return tf.contrib.rnn.MultiRNNCell(cells)

    def _create_placeholder(self):
        self.labels_ = tf.placeholder(tf.float32, shape=(None, self.DECODER_SEQ_LENGTH, self.VOL_SIZE))
        with tf.variable_scope("encoder") as scope:
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, self.ENCODER_SEQ_LENGTH), name="encoder_inputs")
        with tf.variable_scope("decoder") as scope:
            self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, self.DECODER_SEQ_LENGTH), name="decoder_inputs")

    def init_state(self, cell, batch_size):
        if self.TRAINABLE:
            return cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        else:
            return cell.zero_state(batch_size=1, dtype=tf.float32)

    def variable(self, shape, name):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name=name)

    def _inference(self):

        self.embedding = tf.get_variable(
            "embedding", [self.VOL_SIZE, self.EMBEDDING_SIZE], dtype=tf.float32)
        num_classes = self.VOL_SIZE
        ## use softmax to map decoder_output to number(0-5,EOS)
        self.softmax_w = self.variable(
            name="softmax_w", shape=[self.HIDDEN_UNIT, num_classes])
        self.softmax_b = self.variable(name="softmax_b", shape=[num_classes])

        with tf.variable_scope("encoder") as scope:
            encoder_embedding_vectors = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            self.encoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            encoder_state = self.init_state(self.encoder_cell, self.batch_size)
            for time_step in xrange(self.ENCODER_NUM_STEPS):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                encoder_output, encoder_state = self.encoder_cell(encoder_embedding_vectors[:, time_step, :], encoder_state)

        last_encoder_state = encoder_state

        self.decoder_outputs = []
        self.internal = []
        with tf.variable_scope("decoder") as scope:
            decoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            if self.TRAINABLE:
                decoder_embedding_vectors = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
                for time_step in xrange(self.DECODER_NUM_STEPS):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        decoder_state = last_encoder_state
                    decoder_output, decoder_state = decoder_cell(decoder_embedding_vectors[:, time_step, :], decoder_state)
                    self.decoder_outputs.append(decoder_output)
            else:
                gen_decoder_input = tf.constant(self.EOS, shape=(1, 1), dtype=tf.int32)
                for time_step in xrange(self.DECODER_NUM_STEPS):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        decoder_state = last_encoder_state
                    print('step:', time_step)
                    gen_decoder_input_vector = tf.nn.embedding_lookup(self.embedding, gen_decoder_input)
                    decoder_output, decoder_state = decoder_cell(gen_decoder_input_vector[:,0,:], decoder_state)
                    index = self._neural_decoder_output_index(decoder_output)
                    gen_decoder_input = tf.reshape(index,[-1, 1])
                    self.decoder_outputs.append(decoder_output)
                    self.internal.append(gen_decoder_input)
                # logits_series = tf.matmul(decoder_output, softmax_w) + softmax_b  # Broadcasted addition
                # # logits_series = tf.nn.softmax(logits_series, dim=1)
                # y_ = labels_[:, time_step, :]

    ## map decoder_output back to decoder_input(the index)
    ## this function is used when decoder inputs aren't given
    def _neural_decoder_output_index(self, decoder_output):
        num_classes = self.VOL_SIZE
        logits_series = tf.matmul(decoder_output, self.softmax_w) + self.softmax_b
        probs = tf.reshape(tf.nn.softmax(logits_series), [-1, 1])
        index = tf.argmax(probs)
        return index

    def _create_loss(self):
        self.loss = 0
        self.logits_ = []

        for time_step in xrange(self.DECODER_NUM_STEPS):
            decoder_output = self.decoder_outputs[time_step]
            logits_series = tf.matmul(decoder_output, self.softmax_w) + self.softmax_b  # Broadcasted addition
            self.logits_.append(tf.nn.softmax(logits_series))
            y_ = self.labels_[:, time_step, :]
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_series))
            self.loss = self.loss + cross_entropy
        self.predictions_ = [tf.argmax(logit, axis=1) for logit in self.logits_]

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_loss()
        self._create_optimizer()