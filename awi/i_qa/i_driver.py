#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

See README.md to learn what this code has done!

Also SEE https://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm
for special treatment for this code
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
from tensorflow.python import debug as tf_debug

from i_config import Config
from i_dual import IDual

reload(sys)
sys.setdefaultencoding("utf-8")

def train():
    data_helper = Config('../../data/classified/qa/qa_small.txt')
    model = IDual(trainable=True, data_helper=data_helper)
    # model.build_graph()
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        gen = data_helper.generate_batch_data(32)
        for query_seqs, query_lens, response_seqs, response_lens in gen:
            batch = [query_seqs, query_lens, response_seqs, response_lens]
            # opts, feed_dict = step(batch=batch, model=model, is_training=True)
            opts = [model.optOp, model.mean_loss, model.training_summaries]
            _, loss, _ = sess.run(opts, feed_dict={model.query_seqs.name: batch[0], \
                                                   model.query_length.name: batch[1], \
                                                   model.response_seqs.name: batch[2], \
                                                   model.response_length.name: batch[3],
                                                   model.labels.name: np.eye(len(batch[0]))})
            print(loss)

if __name__ == '__main__':
    train()