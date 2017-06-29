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

This file contains the code to build the model

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import time
import csv
import sys
import numpy as np
import tensorflow as tf
import inspect
import jieba
import _uniout
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle

reload(sys)
sys.setdefaultencoding("utf-8")


class NBSeqClassifier:

    def __init__(self, data_path):
        print('initilizing classifier...')
        self.data_path = data_path
        self.num_vol = 0
        self.vol = {}
        self.classes = {}
        self.index_classes = {}
        self.classes_num_sub = {}
        self.classifiers = {}

    def build(self):
        index = 0
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                index = index + 1

                a = line[0].encode('utf-8')
                b = line[1].encode('utf-8')
                last_slot, slot = a.split(",")
                # print("a:{},b:{}".format(a, b))
                if last_slot not in self.classes:
                    self.classes[last_slot] = {}
                    self.index_classes[last_slot] = {}
                    self.classes_num_sub[last_slot] = 1
                    self.classes.get(last_slot)[slot] = 1
                    self.index_classes.get(last_slot)[1] = slot
                else:
                    if slot not in self.classes.get(last_slot):
                        self.classes.get(last_slot)[
                            slot] = self.classes_num_sub.get(last_slot) + 1
                        self.index_classes.get(last_slot)[
                            self.classes_num_sub.get(last_slot) + 1] = slot
                        self.classes_num_sub[
                            last_slot] = self.classes_num_sub.get(last_slot) + 1

                if last_slot != "ROOT":
                    ax = self.cut(last_slot)
                    self.into_vol(ax)

                ax = self.cut(slot)
                self.into_vol(ax)

                bx = self.cut(b)
                self.into_vol(bx)
        print("self.vol:", _uniout.unescape(str(self.vol), 'utf8'))
        print('**************************************************')
        print("self.classes:", _uniout.unescape(str(self.classes), 'utf8'))
        print('**************************************************')
        print("self.index_classes:", _uniout.unescape(
            str(self.index_classes), 'utf8'))
        print('**************************************************')
        print("self.classes_num_sub:", _uniout.unescape(
            str(self.classes_num_sub), 'utf8'))

        # for key, value in self.vol.iteritems():
        #     print(key, value)
        # print(len(self.vol))
        # print([v for v in sorted(self.vol.values())])

    def train_classifier(self):
        embeddings = {}
        classes = {}
        weights = {}
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                last_slot, slot = key.split(',')
                tokens = self.cut(input_)
                embedding = self.sequence_to_embedding(tokens)
                if last_slot not in embeddings:
                    embeddings[last_slot] = []
                embeddings[last_slot].append(embedding)

                # print(last_slot, slot)
                if last_slot not in classes:
                    classes[last_slot] = []
                cls = self.classes[last_slot][slot]
                classes[last_slot].append(cls)

                if last_slot not in weights:
                    weights[last_slot] = []
                w = float(line[2])
                weights[last_slot].append(w)
        print("classes:", _uniout.unescape(str(classes), 'utf8'))

        for key, embs in embeddings.iteritems():
            embeddings[key] = np.array(embs)
        for key, cs in classes.iteritems():
            classes[key] = np.array(cs)
        for key, ww in weights.iteritems():
            weights[key] = np.array(ww)

        for i, last_slot in enumerate(classes.keys()):
            if self.classes_num_sub[last_slot] > 1:
                print(i)
                clf = GradientBoostingClassifier(n_estimators=1000)
                # clf = MultinomialNB(
                #     alpha=0.01, class_prior=None, fit_prior=True)
                clf.fit(embeddings[last_slot], classes[
                        last_slot])
                self.classifiers[last_slot] = clf

        # test
        input_ = '办卡'
        print(self.predict('ROOT', input_))

    def into_vol(self, tokens):
        _tokens = tokens.split("#")
        for token in _tokens:
            if token == "," or token == "?" or token == "":
                continue
            if token not in self.vol:
                self.vol[token] = self.num_vol
                self.num_vol = self.num_vol + 1

    def sequence_to_embedding(self, tokens):
        vector = np.zeros(self.num_vol)
        _tokens = tokens.split("#")
        for token in _tokens:
            if token == ',' or token == '?' or token == '':
                continue
            index = self.vol[token]
            vector[index] = 1
        return vector

    def cut(self, input_):
        # return self.tokenize(input_)
        return self.jieba_cut(input_)

    def jieba_cut(self, input_):
        seg = "#".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def tokenize(self, input_):
        s = input_
        L = []
        for ch in s:
            L.append(ch)

        return "#".join(L)

    def predict(self, parent_slot, input_):
        tokens = self.cut(input_)
        embeddings = [self.sequence_to_embedding(tokens)]
        clf = self.classifiers[parent_slot]
        class_ = clf.predict(embeddings)
        for c in class_:
            return self.index_classes[parent_slot][c]

    def test(self, model_path):
        correct = 0.0
        total = 0.0
        with open(model_path, "rb") as input_file:
            # test
            with open(self.data_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    key = line[0].encode('utf-8')
                    input_ = line[1].encode('utf-8')
                    last_slot, slot = key.split(',')
                    if self.classes_num_sub[last_slot] == 1:
                        continue
                    prediction = self.predict(last_slot, input_)
                    if prediction == slot:
                        correct = correct + 1
                    else:
                        print(input_, last_slot, slot, prediction)
                    total = total + 1
        print('accuracy:' + str(correct / total))


def test_main():
    clf = NBSeqClassifier("../data/train_key.txt")
    clf.build()
    clf.train_classifier()

if __name__ == "__main__":
    clf = NBSeqClassifier("../data/train_pruned.txt")
    clf.build()
    clf.train_classifier()
    with open("../model/seq_clf.pkl", 'wb') as pickle_file:
        pickle.dump(clf, pickle_file, pickle.HIGHEST_PROTOCOL)

    with open("../model/seq_clf.pkl", "rb") as input_file:
        _clf = pickle.load(input_file)
        input_ = '开户'
        print(_clf.predict('ROOT', input_))

        _clf.test("../data/train_pruned.txt")

    # test_main()