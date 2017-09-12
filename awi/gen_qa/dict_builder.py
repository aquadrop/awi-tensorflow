#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A neural chatbot using sequence to sequence model with

Data of Chinese community question-answer pairs and knowledge base (triples):

1. CQA-triple data: cqa_triple.tar.gz
   Tab-delimited format: Question, Answer, Subject, Predicate, Object, ConfidenceScore

2. Triple data: triple.tar.gz
   Tab-delimited format: Subject, Predicate, Object, ConfidenceScore

If you have any question regarding the data, please contact Xin Jiang (Jiang.Xin@huawei.com).

Please use the data only for research purpose. Thanks.

Reference:
Yin, J., Jiang, X., Lu, Z., Shang, L., Li, H., & Li, X. (2015). Neural Generative Question Answering. arXiv preprint arXiv:1512.01337.


"""
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import isfile, join

import sys
import numpy as np
import json


import argparse
import os
import random
import time
import inspect

import numpy as np

import jieba

reload(sys)
sys.setdefaultencoding("utf-8")


def build_question_dict(path, triple, output):
    """
    :param path:
    :return:
    cut question, add predicates
    """
    words = dict()
    parts = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith("part")]
    for part in parts:
        with open(join(path, part), 'r') as f:
            print('processing file..', part)
            for line in f:
                question = line.split("\t")[0]
                word_list = jieba.cut(question)
                for w in word_list:
                    w.replace(" ", "").replace("\t", "")
                    if w:
                        if w in words:
                            words[w] = words[w] + 1
                        else:
                            words[w] = 1

    parts = [f for f in listdir(triple) if isfile(join(triple, f)) and f.startswith("part")]
    for part in parts:
        with open(join(path, part), 'r') as f:
            print('processing file..', part)
            for line in f:
                question = line.split("\t")[1]
                word_list = jieba.cut(question)
                for w in word_list:
                    w.replace(" ", "").replace("\t", "")
                    if w:
                        if w in words:
                            words[w] = words[w] + 1
                        else:
                            words[w] = 1

    tuples = sorted(words.items(), key=lambda x:x[1], reverse=True)
    with open(output, 'w') as f:
        for key, value in tuples:
            f.write(key + "\t" + str(value) + "\n")


def build_dict(path, triple, output):
    """
    :param path:
    :return:
    cut question, add predicates
    """
    words = dict()
    parts = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith("part-00000")]
    for part in parts:
        with open(join(path, part), 'r') as f:
            print('processing file..', part)
            for line in f:
                question = line.split("\t")[0]
                word_list = ' '.join(jieba.cut(question)).split(" ")
                answer = line.split("\t")[1]
                word_list.extend(' '.join(jieba.cut(answer)).split(" "))
                for w in word_list:
                    w.replace(" ", "").replace("\t", "")
                    if w:
                        if w in words:
                            words[w] = words[w] + 1
                        else:
                            words[w] = 1

    parts = [f for f in listdir(triple) if isfile(join(triple, f)) and f.startswith("part-00000")]
    for part in parts:
        with open(join(path, part), 'r') as f:
            print('processing file..', part)
            for line in f:
                entity = line.split("\t")[0]
                word_list = ' '.join(jieba.cut(entity)).split(" ")
                predicate = line.split("\t")[1]
                word_list.extend(' '.join(jieba.cut(predicate)).split(" "))
                subject = line.split("\t")[2]
                word_list.extend(' '.join(jieba.cut(subject)).split(" "))
                for w in word_list:
                    w.replace(" ", "").replace("\t", "")
                    if w:
                        if w in words:
                            words[w] = words[w] + 1
                        else:
                            words[w] = 1

    tuples = sorted(words.items(), key=lambda x:x[1], reverse=True)
    # #PAD#
    # #UNK#
    # #GO#
    # #EOS#
    tuples.insert(0, ("#PAD#", 1))
    tuples.insert(0, ("#UNK#", 1))
    tuples.insert(0, ("#GO#", 1))
    tuples.insert(0, ("#EOS#", 1))

    with open(output, 'w') as f:
        for key, value in tuples:
            f.write(key + "\t" + str(value) + "\n")


def build_answer_dict(path, triple, output):
    """
        :param path:
        :return:
        cut question, add predicates
        """
    words = dict()
    parts = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith("part")]
    for part in parts:
        with open(join(path, part), 'r') as f:
            print('processing file..', part)
            for line in f:
                answer = line.split("\t")[1]
                word_list = jieba.cut(answer)
                for w in word_list:
                    w.replace(" ", "").replace("\t", "")
                    if w:
                        if w in words:
                            words[w] = words[w] + 1
                        else:
                            words[w] = 1

    tuples = sorted(words.items(), key=lambda x: x[1], reverse=True)
    with open(output, 'w') as f:
        for key, value in tuples:
            f.write(key + "\t" + str(value) + "\n")

if __name__ == '__main__':
    # build_question_dict("../../data/huawei/cqa_triple_match",
    #                     "../../data/huawei/triple",
    #                     "../../data/huawei/dict/q_dict.txt")
    # build_answer_dict("../../data/huawei/cqa_triple_match",
    #                     "../../data/huawei/triple",
    #                     "../../data/huawei/dict/a_dict.txt")
    build_dict("../../data/huawei/cqa_triple_match",
               "../../data/huawei/triple",
               "../../data/huawei/dict/dict.txt")
