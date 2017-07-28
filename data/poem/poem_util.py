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

This file contains the hyperparameters for the model.

See readme.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset

import csv
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# with open('poem_raw.txt', 'rb') as f:
#     with open('poem.txt', 'w') as wf:
#         for line in f:
#             line = line.decode('utf-8')
#             if u'##' in line:
#                 sentences = line.split("##")
#                 for st in sentences:
#                     st = st.strip()
#                     if len(st) == 5:
#                         wf.write(str(st.strip()) + '\n')
#             if u'◎卷' in line:
#                 wf.write('\n')

with open('./poem.txt', 'r') as f:
    dictionary = set()
    for line in f:
        line = line.decode('utf-8')
        for cc in line:
            # cc = cc.encode('utf-8')
            dictionary.add(cc)

    print(len(dictionary))
