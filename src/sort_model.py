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
    batch_size = 64

    EOS = 6
    VOL_SIZE = 7
    EMBEDDING_SIZE = 10
    ENCODER_SEQ_LENGTH = 3
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH = ENCODER_SEQ_LENGTH + 1  ## plus 1 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

    HIDDEN_UNIT = 128
    N_LAYER = 3


    def __init__(self):
        print('initilizing model...')

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

    @staticmethod
    def init_state(cell, batch_size):
        return cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    def _inference(self):
        self.embedding = tf.get_variable(
            "embedding", [self.VOL_SIZE, self.EMBEDDING_SIZE], dtype=tf.float32)

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
        with tf.variable_scope("decoder") as scope:
            decoder_embedding_vectors = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
            decoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            for time_step in xrange(self.DECODER_NUM_STEPS):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                else:
                    decoder_state = last_encoder_state
                decoder_output, decoder_state = decoder_cell(decoder_embedding_vectors[:, time_step, :], decoder_state)
                self.decoder_outputs.append(decoder_output)

                # logits_series = tf.matmul(decoder_output, softmax_w) + softmax_b  # Broadcasted addition
                # # logits_series = tf.nn.softmax(logits_series, dim=1)
                # y_ = labels_[:, time_step, :]



    def _create_optimizer(self):
        self.loss = 0
        num_classes = self.VOL_SIZE
        ## use softmax to map decoder_output to number(0-5,EOS)
        softmax_w = tf.get_variable(
            "softmax_w", [self.HIDDEN_UNIT, num_classes], tf.float32)
        softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
        for time_step in xrange(self.DECODER_SEQ_LENGTH):
            decoder_output = self.decoder_outputs[time_step]
            logits_series = tf.matmul(decoder_output, softmax_w) + softmax_b  # Broadcasted addition
            y_ = self.labels_[:, time_step, :]
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_series))
            self.loss = self.loss + cross_entropy
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_optimizer()