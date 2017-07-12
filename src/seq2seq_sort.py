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

def sort_op_data(size = SortModel.batch_size):
    while True:
        encoder_inputs = []
        decoder_inputs = []
        labels = []  ## this is basically identical to decoder inputs tailed with EOS
        for i in xrange(size):
            _input, _output, label = gen_triple()
            encoder_inputs.append(_input)
            decoder_inputs.append(_output)
            labels.append(label)

        yield np.asarray(encoder_inputs, dtype=np.int32).reshape(size, SortModel.ENCODER_NUM_STEPS), np.asarray(\
            decoder_inputs, dtype=np.int32).reshape((size, SortModel.DECODER_NUM_STEPS)), np.asarray(labels)

def train():

    model = SortModel()
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        gen = sort_op_data()
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        i = 0
        for e_inputs, d_inputs, labels in gen:
            model.optimizer.run(feed_dict={model.encoder_inputs.name: e_inputs,\
                                           model.decoder_inputs.name: d_inputs,\
                                           model.labels_.name: labels})
            if (i + 1) % 100 == 0:
                loss = sess.run(model.loss, feed_dict={model.encoder_inputs.name: e_inputs,
                    model.decoder_inputs.name: d_inputs,
                    model.labels_.name: labels})
                print(loss)
                if loss < 10:
                    # print(sess.run(model.logits_, feed_dict={model.encoder_inputs.name: e_inputs,
                    #                                       model.decoder_inputs.name: d_inputs,
                    #                                       model.labels_.name: labels}))
                    print(sess.run(model.encoder_inputs, feed_dict={model.encoder_inputs.name: e_inputs,\
                                                             model.decoder_inputs.name: d_inputs,\
                                                             model.labels_.name: labels})[0])
                    print(sess.run(model.decoder_inputs, feed_dict={model.encoder_inputs.name: e_inputs,\
                                                                    model.decoder_inputs.name: d_inputs,\
                                                                    model.labels_.name: labels})[0])
                    predictions = np.array(sess.run(model.predictions_, feed_dict={model.encoder_inputs.name: e_inputs,\
                                                                    model.decoder_inputs.name: d_inputs,\
                                                                    model.labels_.name: labels}))
                    softmax_w = sess.run(model.softmax_w, feed_dict={model.encoder_inputs.name: e_inputs,\
                                                                    model.decoder_inputs.name: d_inputs,\
                                                                    model.labels_.name: labels})[0]
                    predictions = predictions.reshape([-1, 4])[0]
                    print("predictions:", predictions, softmax_w)
                saver.save(sess, "../model/rnn/rnn", global_step=i)
            i = i + 1

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("../model/rnn/rnn"))
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
    model = SortModel(False)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        while True:
            line = _get_user_input()
            # gen = sort_op_data(1)
            e_inputs = list()
            e_inputs.append(np.array([int(x) for x in line.split(",")]))
            e_inputs = np.array(e_inputs)
            predictions = np.array(np.array(sess.run(model.predictions_, feed_dict={model.encoder_inputs.name: e_inputs}))).reshape([-1, 4])
            print(e_inputs, predictions)

    return 0

if __name__ == "__main__":
    # train()
    run_sort()
    # a = np.random.rand(2,2)
    # x = tf.placeholder(tf.float32, shape=(2, 2))
    # y = tf.matmul(x, x)
    # with tf.Session() as sess:
    #     print(sess.run(y, feed_dict={x:a}))
