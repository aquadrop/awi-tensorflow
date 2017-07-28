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
import json

import uuid

import re

import unicodedata

reload(sys)
sys.setdefaultencoding("utf-8")

topic_sign = ['一.','二.','三.','四.','五.','六.','七.']
talk_sign = r'^[0-9]+.*$'
talk_pattern = re.compile(talk_sign)
guest_sign = r'G:.*'
guest_pattern = re.compile(guest_sign)
bot_sign = r'B:.*'
bot_pattern = re.compile(bot_sign)

GUEST_START = "AUTOMATA_WAKE_UP"

def get_topic(line):
    tt = []
    start = False
    for c in line:
        if c == ':':
            start = True
            continue
        if start:
            tt.append(c)
    return ''.join(tt)

def topic_start(line):
    return '话题:' in line

def write_session(f, session):
    combines = combination(session)
    for cmb in combines:
        for s in cmb:
            f.write(s + '\n')
        f.write('\n')

def combination(session, index = 0):
    if len(session) == 0:
        return []
    if index == len(session) - 1:
        start = [[ss] for ss in session[index]]
        return start
    sub_c = combination(session, index= index + 1)
    concat = []
    for sentence in session[index]:
        for branch in sub_c:
            branch.insert(0, sentence)
            concat.append(branch)
    return concat


def interactive(file_, write_file_):
    with open(write_file_, 'w') as wf:
        with open(file_, 'rb') as f:
            session = []
            session_turn = 0
            for line in f:
                line = line.strip().decode('utf-8')
                line = unicodedata.normalize('NFKC', line).replace('\t', '').replace(' ', '')
                if topic_start(line):
                    continue
                if talk_pattern.match(str(line)):
                    write_session(wf, session)
                    session = []
                    session_turn = -1
                    line = re.sub('^[0-9]+(.)', '', str(line)).strip()
                    if guest_pattern.match(str(line)):
                        guest_line = str(line).replace('G:', '')
                        if len(session) % 2 == 0:
                            session.append([guest_line])
                            session_turn += 1
                        else:
                            session[session_turn].append(guest_line)
                    elif bot_pattern.match(str(line)):
                        if (len(session) == 0):
                            session.append([GUEST_START])
                            session_turn += 1
                        bot_line = str(line).replace('B:', '')
                        if len(session) % 2 == 1:
                            session.append([bot_line])
                            session_turn += 1
                        else:
                            session[session_turn].append(bot_line)
                    else:
                        print(line)
                        guest_line = str(line).replace('G:', '')
                        if len(session) % 2 == 0:
                            session.append([guest_line])
                            session_turn += 1
                        else:
                            session[session_turn].append(guest_line)
                    continue
                if guest_pattern.match(str(line)):
                    guest_line = str(line).replace('G:', '')
                    if len(session) % 2 == 0:
                        session.append([guest_line])
                        session_turn += 1
                    else:
                        session[session_turn].append(guest_line)
                if bot_pattern.match(str(line)):
                    bot_line = str(line).replace('B:', '')
                    if len(session) % 2 == 1:
                        session.append([bot_line])
                        session_turn += 1
                    else:
                        session[session_turn].append(bot_line)

if __name__ == '__main__':
	# interactive('../data/interactive/整理后的客服接待语料.txt','../data/interactive/interactive-all.json')
	interactive('../data/classified/interactive/2017互动话术汇总版4.10.txt','../data/classified/interactive/interactive2017.txt')