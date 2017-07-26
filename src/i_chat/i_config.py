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

    TURN_NUM = 4

    PAD = 2
    EOS = 1
    GO = 0

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
        self.sessions = []
        lines = []
        with open(file_, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip('\n')
                lnum += 1
                lines.append(line)
                if not line:
                    self.sessions.append(lines)
                    lines = []
                    continue
                for cc in line:
                    # cc = cc.encode('utf-8')
                    set_.add(cc)

            print('built size of ', len(set_), ' dictionary', lnum)
        self.chars = []
        self.dic = {}

        self.chars.append(self.GO)
        self.dic[self.GO] = self.GO_
        self.chars.append(self.EOS)
        self.dic[self.EOS] = self.EOS_
        self.chars.append(self.PAD)
        self.dic[self.PAD] = self.PAD_

        index = 3
        for char in set_:
            self.chars.append(char)
            self.dic[char] = index
            index = index + 1

        self.VOL_SIZE = len(self.chars)

    turn_round = 0
    checkpoint = 0
    moving_checkpoint = 0

    def generate_batch_data(self, batch_size=32):
        batch_encoder_inputs = list()
        batch_decoder_inputs = list()
        labels = list()

        for i in xrange(batch_size):
            session = self.sessions[self.checkpoint + i]
            if (len(session) < self.TURN_NUM):
                self.checkpoint += 1
                session = self.sessions[self.checkpoint + i]

            batch_encoder_inputs.append(self.translate(session[self.turn_round]))
            target = np.array(self.translate(session[self.turn_round + 1]))
            decoder_input = np.append(target, self.EOS)
            label = np.append(self.GO, target)
            batch_decoder_inputs.append(decoder_input)
            labels.append(label)

            self.moving_checkpoint = self.checkpoint + i

        self.turn_round += 2
        if self.turn_round == self.TURN_NUM:
            self.turn_round = 0
        self.checkpoint = self.moving_checkpoint
        yield np.array(batch_encoder_inputs), np.array(batch_decoder_inputs), np.array(labels)


    def translate(self, sentence):
        indices = list()
        for c in sentence:
            tr = self.dic[c]
            indices.append(tr)
        return indices

    def recover(self, index):
        sentence = []
        for ii in index:
            sentence.append(str(self.chars[int(ii)]))
        return ''.join(sentence)

if __name__ == '__main__':
    config = Config('../../data//business/business_sessions.txt')
    # with open('../../data/log.txt', 'w') as log_:
    batch_size = 2
    for a, b, c in config.generate_batch_data(batch_size):
        for i in xrange(len(a)):
            print(config.recover(a[i]), config.recover(b[i]), config.recover(c[i]))
        print('===========')