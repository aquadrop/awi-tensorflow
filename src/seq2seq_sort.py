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

from sort_model import SortModel

def one_hot_triple(index):
    zeros_a = np.zeros(SortModel.VOL_SIZE, dtype=np.float32)
    zeros_a[index] = 1
    return zeros_a


def gen_triple():
    _input = np.random.random_integers(5, size=SortModel.ENCODER_SEQ_LENGTH)
    _output = np.sort(_input)

    label = []
    for o in _output:
        label.append(one_hot_triple(o))
    label.append(one_hot_triple(SortModel.EOS))

    _output = np.append(SortModel.EOS, _output)
    return _input, _output, label


def plus_op_data():
    while True:
        encoder_inputs = []
        decoder_inputs = []
        labels = []  ## this is basically identical to decoder inputs tailed with EOS
        for i in xrange(SortModel.batch_size):
            _input, _output, label = gen_triple()
            encoder_inputs.append(_input)
            decoder_inputs.append(_output)
            labels.append(label)

        yield np.asarray(encoder_inputs, dtype=np.int32).reshape(SortModel.batch_size, SortModel.ENCODER_SEQ_LENGTH), np.asarray(
            decoder_inputs, dtype=np.int32).reshape((SortModel.batch_size, SortModel.DECODER_SEQ_LENGTH)), np.asarray(labels)


def train():

    model = SortModel()
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        gen = plus_op_data()
        sess.run(tf.global_variables_initializer())
        i = 0
        for e_inputs, d_inputs, labels in gen:
            model.optimizer.run(feed_dict={model.encoder_inputs.name: e_inputs,
                                           model.decoder_inputs.name: d_inputs,
                                           model.labels_.name: labels})
            if (i + 1) % 100 == 0:
                print(sess.run(model.loss, feed_dict={model.encoder_inputs.name: e_inputs,
                    model.decoder_inputs.name: d_inputs,
                    model.labels_.name: labels}))
                saver.save(sess, "../model/model")
            i = i + 1


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("../model"))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the SortBot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the SortBot")

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def run_sort():
    model = SortModel()
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        _check_restore_parameters(sess, saver)
        while True:
            line = _get_user_input()
            _input = [int(x) for x in line.split(",")]
            _input = np.asarray(_input).reshape(1, 3)
            decoder_outputs = sess.run(model.decoder_outputs, feed_dict={model.encoder_inputs.name: _input})
            print(decoder_outputs)

    return 0


if __name__ == "__main__":
    # train()
    run_sort()
    # a = np.random.rand(2,2)
    # x = tf.placeholder(tf.float32, shape=(2, 2))
    # y = tf.matmul(x, x)
    # with tf.Session() as sess:
    #     print(sess.run(y, feed_dict={x:a}))
