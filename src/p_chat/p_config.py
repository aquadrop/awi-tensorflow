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
                if not line:
                    continue
                self.lines.append(line)
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
        self.chars.append('#EOS#')
        self.dic['#EOS#'] = index

        self.VOL_SIZE = len(self.chars)
        self.EOS = self.VOL_SIZE - 1

    def gen_triple(self, check = 0):
        if check >= len(self.lines) - 1:
            check = 0
        source = self.lines[check]
        target = self.lines[check + 1]
        check =  check + 2
        source = source.decode('utf-8')
        target = target.decode('utf-8')
        source_index = []
        for c in source:
            source_index.append(self.dic[c])
        target_index = []
        label_index = []
        for c in target:
            target_index.append(self.dic[c])
            label_index.append(self.dic[c])

        encoder_input = np.array(source_index)
        decoder_input = np.array(target_index)
        labels = np.array(label_index)

        ## append EOS to decoder
        decoder_input = np.append(self.EOS, decoder_input)
        labels = np.append(labels, self.EOS)
        # print(self.recover(encoder_input), self.recover(decoder_input), self.recover(labels), check, len(self.lines))
        return check, encoder_input, decoder_input, labels

    def get_batch_data(self, batch_size):
        check = 0
        while True:
            encoder_inputs = []
            decoder_inputs = []
            labels = []
            for bb in xrange(batch_size):
                pad = bb * (batch_size - 1) * 2
                check, a, b, c = self.gen_triple(check + pad)
                encoder_inputs.append(a)
                decoder_inputs.append(b)
                labels.append(c)
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = np.array(decoder_inputs)
            labels = np.array(labels)
            yield encoder_inputs, decoder_inputs, labels

    def recover(self, index):
        sentence = []
        for ii in index:
            sentence.append(self.chars[int(ii)])
        return ''.join(sentence)

if __name__ == '__main__':
    config = Config('../../data/poem.txt')
    # with open('../../data/log.txt', 'w') as log_:
    for a, b, c in config.get_batch_data(2):
        print(config.recover(a[0]) + config.recover(b[0]) + config.recover(c[0]))