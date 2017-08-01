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


import sys
import numpy as np
import json


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

    TURN_NUM = 0

    PAD_ = 2
    EOS_ = 1
    GO_ = 0
    UNK_ = 3

    PAD = '#PAD#'
    EOS = '#EOS#'
    GO = '#GO#'
    UNK = '#UNK#'

    def __init__(self, file_, char2index_path, index2char_path):
        self.file_ = file_
        self.char2index_path = char2index_path
        self.index2char_path = index2char_path
        self.char2index_dict = dict()
        self.index2char_dict = dict()
        self._build_()

    def _build_(self):
        self._build_dict()
        self._build_sessions()

    def _build_dict(self):
        print('Building dict...')

        def int2str_key(dic):
            new_dict = dict()
            for key in dic.keys():
                new_dict[int(key)] = dic[key]
            return new_dict

        char2index_f = open(self.char2index_path, 'r')
        index2char_f = open(self.index2char_path, 'r')

        self.char2index_dict = json.load(char2index_f)
        index2char_dict = json.load(index2char_f)

        self.index2char_dict = int2str_key(index2char_dict)

        char2index_f.close()
        index2char_f.close()

        self.VOL_SIZE = len(self.index2char_dict)

    def _build_sessions(self):
        print('Building sessions...')
        self.sessions = []
        lines = []
        with open(self.file_, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip('\n')
                if line:
                    lines.append(line)
                if not line:
                    self.sessions.append(lines)
                    lines = []
                    continue

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

            if self.yield_flag:
                batch_encoder_inputs = list()
                batch_decoder_inputs = list()
                labels = list()

                batch_encoder_inputs_length = list()
                batch_decoder_inputs_length = list()

                for ii in xrange(batch_size):
                    session = self.session_batch[ii]

                    source = self.translate(session[self.turn_round])
                    # print('source:', source)
                    batch_encoder_inputs.append(source)
                    batch_encoder_inputs_length.append(len(source))

                    target = self.translate(
                        session[self.turn_round + 1])
                    decoder_input = [self.GO_] + target
                    batch_decoder_inputs_length.append(len(decoder_input))

                    label = target + [self.EOS_]

                    batch_decoder_inputs.append(decoder_input)
                    labels.append(label)

                self.turn_round += 2

                if self.turn_round == self.TURN_NUM:
                    self.turn_round = 0
                    self.yield_flag = False

                batch_encoder_inputs = self.padding(batch_encoder_inputs)
                batch_decoder_inputs = self.padding(batch_decoder_inputs)
                labels = self.padding(labels)

                yield batch_encoder_inputs, batch_decoder_inputs, labels,\
                    np.array(batch_encoder_inputs_length), np.array(
                        batch_decoder_inputs_length)
            else:
                i = 0
                # print('start turn_num:', len(
                # self.sessions[self.checkpoint + i]))
                self.TURN_NUM = len(self.sessions[self.checkpoint + i])
                if self.TURN_NUM % 2 and self.TURN_NUM not in [7, 9, 11]:
                    self.TURN_NUM += 1
                elif self.TURN_NUM % 2 and self.TURN_NUM in [7, 9, 11]:
                    self.TURN_NUM = 4

                # self.TURN_NUM = 2 if self.TURN_NUM > 4 else self.TURN_NUM
                self.session_batch = list()
                # print('self.turn_num:', self.TURN_NUM)
                while i < batch_size:
                    if (self.checkpoint + i >= len(self.sessions)):
                        self.checkpoint = -i
                    session = self.sessions[self.checkpoint + i]

                    if (len(session) < self.TURN_NUM):
                        self.checkpoint += 1
                        continue
                    i += 1
                    self.session_batch.append(session)

                    if (self.checkpoint + i >= len(self.sessions)):
                        self.checkpoint = -i

                self.checkpoint = self.checkpoint + i
                self.yield_flag = True

    def translate(self, sentence):
        indices = list()
        for c in sentence:
            tr = self.char2index_dict.get(c, 3)
            indices.append(tr)
        return indices

    def recover(self, index):
        sentence = []
        for ii in index:
            if ii == self.GO_ or ii == self.PAD_ or ii == -1:
                continue
            if ii == self.EOS_:
                break
            sentence.append(str(self.chars[int(ii)]))
        return ''.join(sentence)


if __name__ == '__main__':
    config = Config('../../data/classified/interactive/interactive.txt',
                    '../../data/char_table/char2index_dict_small.txt', '../../data/char_table/index2char_dict_small.txt')
    # with open('../../data/log.txt', 'w') as log_:
    batch_size = 1
    for a, b, c, d, e in config.generate_batch_data(batch_size):
        for i in xrange(len(a)):
            print(config.recover(a[i]), config.recover(
                b[i]), config.recover(c[i]))
        print('===========')
