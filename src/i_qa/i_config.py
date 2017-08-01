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
    DECODER_SEQ_LENGTH = ENCODER_NUM_STEPS + 1  # plus 1 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

    SPLITTER = '\t'

    TURN_NUM = 4

    PAD_ = 2
    EOS_ = 1
    GO_ = 0

    PAD = '#PAD#'
    EOS = '#EOS#'
    GO = '#GO#'

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
    yield_flag = False
    session_batch = list()

    def padding(self, inputs):
        batch_size = len(inputs)
        sequence_lengths = [len(seq) for seq in inputs]
        max_sequence_length = max(sequence_lengths)

        inputs_batch_major = np.ones(
            (batch_size, max_sequence_length), np.int32) * self.PAD_
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i][j] = element

        return inputs_batch_major

    def generate_batch_data(self, batch_size=32):
        while True:
            query_seqs = list()
            response_seqs = list()
            for i in xrange(batch_size):
                if self.checkpoint + i >= len(self.lines):
                    self.checkpoint = -i
                line = self.lines[self.checkpoint + i]

                compoents = line.split(self.SPLITTER)
                a = compoents[0].replace('\t', '').replace(' ', '')
                b = ''.join(compoents).replace('\t', '').replace(' ', '')

                query = self.translate(a)
                response = self.translate(b)

                query_seqs.append(query)
                response_seqs.append(response)

            query_lens = [len(a) for a in query_seqs]
            response_lens = [len(b) for b in response_seqs]

            query_seqs = self.padding(query_seqs)
            response_seqs = self.padding(response_seqs)

            yield np.array(query_seqs),\
                np.array(query_lens),\
                np.array(response_seqs),\
                np.array(response_lens)


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
    config = Config('../../data/classified/qa/qa.txt')
    # with open('../../data/log.txt', 'w') as log_:
    batch_size = 32
    for a, b, c, d in config.generate_batch_data(batch_size):
        for i in xrange(len(a)):
            print(config.recover(a[i]), b[i], config.recover(c[i]))
        print('===========')
