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

reload(sys)
sys.setdefaultencoding("utf-8")

class Config:

    EMBEDDING_SIZE = 128
    ENCODER_SEQ_LENGTH = 5
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH = ENCODER_NUM_STEPS + 1  ## plus 1 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

    PAD = -1
    EOS = -1
    GO = -1

    PAD_ = '#PAD#'
    EOS_ = '#EOS#'
    GO_ = '#GO#'

    def __init__(self, file_):
        print('building volcabulary...')
        self.file_ = file_
        self._build_dic(file_)

    def _build_dic(self, file_):
        set_ = set()
        lnum = 0
        self.lines = []
        with open(file_, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip('\n')
                lnum += 1
                self.lines.append(line)
                if not line:
                    continue
                for cc in line:
                    # cc = cc.encode('utf-8')
                    set_.add(cc)

            print('built size of ', len(set_), ' dictionary', lnum)
        self.chars = []
        self.dic = {}

        index = 0
        for char in set_:
            self.chars.append(char)
            self.dic[char] = index
            index = index + 1
        self.chars.append('#GO#')
        self.dic['#GO#'] = index
        self.GO = index
        index = index + 1
        self.chars.append('#EOS#')
        self.dic['#EOS#'] = index
        self.EOS = index
        index = index + 1
        self.chars.append('#PAD#')
        self.dic['#PAD#'] = index
        self.PAD = index

        self.VOL_SIZE = len(self.chars)

    def get_session_data(self):
        flag = 0
        while True:
            encoder_inputs = []
            decoder_inputs = []
            labels = []
            for line in self.lines:
                line = line.strip('\n')
                if not line:
                    if len(encoder_inputs) == 0:
                        flag == 0
                        continue
                    encoder_inputs = np.array(encoder_inputs)
                    decoder_inputs = np.array(decoder_inputs)
                    labels = np.array(labels)
                    flag = 0
                    yield len(encoder_inputs), encoder_inputs, decoder_inputs, labels
                    encoder_inputs = []
                    decoder_inputs = []
                    labels = []
                else:
                    encode_input = []
                    decoder_input = []
                    label = []
                    for c in line:
                        index = self.dic[c]
                        if flag % 2 == 0:
                            encode_input.append(index)
                        else:
                            decoder_input.append(index)
                            label.append(index)
                    if flag % 2 == 0:
                        encoder_inputs.append(encode_input)
                    else:
                        decoder_input.insert(0, self.GO)
                        decoder_inputs.append(decoder_input)
                        label.append(self.EOS)
                        labels.append(label)
                    flag += 1


    def recover(self, index):
        sentence = []
        for ii in index:
            sentence.append(self.chars[int(ii)])
        return ''.join(sentence)

if __name__ == '__main__':
    config = Config('../../data/poem.txt')
    # with open('../../data/log.txt', 'w') as log_:
    batch_size = 32
    for l, a, b, c in config.get_session_data():
        for i in xrange(len(a)):
            print(str(l) + config.recover(a[i]) + config.recover(b[i]) + config.recover(c[i]), len(a))
        print('===========')