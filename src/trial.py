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
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import config
import data


class AWI:

    def __init__(self):
        print('initializing...')

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in xrange(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in xrange(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in xrange(config.BUCKETS[-1][1] + 1)]

    def single_cell(self):
      return tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)

    def train(self):
        pass

    def stacked_rnn(self):
        return tf.contrib.rnn.MultiRNNCell([self.single_cell() for _  in range(config.NUM_LAYERS)])

    def init_state(self, cell, batch_size):
        return cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    def _build(self, batch_size):
        self.encoder_cell = self.stacked_rnn()
        self.encoder_state = self.init_state(self.encoder_cell)
        self.intention_cell = self.stacked_rnn()
        self.intention_state = self.init_state(self.intention_cell)
        self.decoder_inputs = self.stacked_rnn()
        self.decoder_state = self.init_state(self.decoder_cell)


    def _run_step(self, encoder_inputs, decoder_inputs):
        ## encoder_cell input
        output, state = self.encoder_cell(encoder_inputs, self.encoder_state)
        self.encoder_state = state
        
        pass

    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(w, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(local_w_t),
                                              biases=local_b,
                                              labels=labels,
                                              inputs=local_inputs,
                                              num_sampled=config.NUM_SAMPLES,
                                              num_classes=config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

    def chat(self):
        pass

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices={'train', 'chat'},
                            default='train', help="mode. if not specified, it's in the train mode")
        args = parser.parse_args()

        if not os.path.isdir(config.PROCESSED_PATH):
            data.prepare_raw_data()
            data.process_data()
        print('Data ready!')
        # create checkpoints folder if there isn't one already
        data.make_dir(config.CPT_PATH)
        if not args.mode:
            args.mode = 'train'
        if args.mode == 'train':
            train()
        elif args.mode == 'chat':
            chat()

if __name__ == "__main__":
    awi = AWI()
    awi.main()