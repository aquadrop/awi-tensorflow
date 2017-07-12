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

class AttentionSortModel:

    batch_size = 1

    VOL_SIZE = 7
    EOS = VOL_SIZE - 1
    EMBEDDING_SIZE = 10
    ENCODER_SEQ_LENGTH = 2
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH =  3  ## plus 0 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH
    # TURN_LENGTH = 3

    HIDDEN_UNIT = 128
    N_LAYER = 3

    TRAINABLE = True


    def __init__(self, trainable = True):
        print('initilizing model...')
        self.TRAINABLE = trainable
        self.turn_index = 0

    def reset_turn(self):
        self.turn_index = 0

    def increment_turn(self):
        self.turn_index = self.turn_index + 1

    def single_cell(self, size=128):
        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)

    def stacked_rnn(self, size=128):
        return tf.contrib.rnn.MultiRNNCell([self.single_cell(size) for _ in range(self.N_LAYER)])
        # cells = list()
        # cells.append(single_cell(size))
        # cells.append(single_cell(size/2))
        # return tf.contrib.rnn.MultiRNNCell(cells)

    def _create_placeholder(self):
        self.labels_ = tf.placeholder(tf.float32, shape=(None, self.DECODER_SEQ_LENGTH, self.VOL_SIZE))
        with tf.variable_scope("encoder") as scope:
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, self.ENCODER_SEQ_LENGTH), name="encoder_inputs")
        with tf.variable_scope("decoder") as scope:
            self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, self.DECODER_SEQ_LENGTH), name="decoder_inputs")

    def init_state(self, cell, batch_size):
        if self.TRAINABLE:
            return cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        else:
            return cell.zero_state(batch_size=1, dtype=tf.float32)

    def variable(self, shape, name):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name=name)

    ## refer HEAVILY to the paper:  https://arxiv.org/pdf/1409.0473.pdf supplementary part
    ## remember! by default RNN state DOES NOT keep in the next batch!!!!!!
    def _inference(self):

        self.embedding = tf.get_variable(
            "embedding", [self.VOL_SIZE, self.EMBEDDING_SIZE], dtype=tf.float32)
        num_classes = self.VOL_SIZE
        ## use softmax to map decoder_output to number(0-5,EOS)
        self.softmax_w = self.variable(
            name="softmax_w", shape=[self.HIDDEN_UNIT, num_classes])
        self.softmax_b = self.variable(name="softmax_b", shape=[num_classes])

        ## prepare to compute c_i = \sum a_{ij}h_j, encoder_states are h_js
        hidden_states = []
        self.W_a = self.variable(name="attention_w_a", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.U_a = self.variable(name="attention_u_a", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.v_a = self.variable(name="attention_v_a", shape=[1, self.EMBEDDING_SIZE])

        ## connect intention with decoder
        ## connect intention with intention
        self.I_E = self.variable(name="intention_e", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.encoder_to_intention_b = self.variable(name="encoder_intention_b", shape=[self.HIDDEN_UNIT])
        self.I_I = self.variable(name="intention_i", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.intention_to_decoder_b = self.variable(name="intention_decoder_b", shape=[self.HIDDEN_UNIT])
        # self.C = self.variable(name="attention_C", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])

        with tf.variable_scope("encoder") as scope:
            encoder_embedding_vectors = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            self.encoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)

            self.encoder_state = self.get_state_variables(self.batch_size, self.encoder_cell)

            for time_step in xrange(self.ENCODER_NUM_STEPS):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                else:
                    encoder_state = self.encoder_state
                encoder_output, encoder_state = self.encoder_cell(encoder_embedding_vectors[:, time_step, :], encoder_state)
                ## can be concat way
                hidden_state = self._concat_hidden(encoder_state) ##
                hidden_states.append(hidden_state) ## steps, (batch, hidden_unit) <-- tensor

        # compute U_a*h_j quote:"this vector can be pre-computed.. U_a is R^n * n, h_j is R^n"
        # U_ah = []
        # for h in hidden_states:
        #     ## h.shape is BATCH, HIDDEN_UNIT
        #     u_ahj = tf.matmul(h, self.U_a)
        #     U_ah.append(u_ahj)

        # hidden_states = tf.stack(hidden_states)
        self.decoder_outputs = []
        # self.internal = []
        #
        with tf.variable_scope("decoder") as scope:
            self.decoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            self.decoder_state = self.get_state_variables(self.batch_size, self.decoder_cell)
        #
        # building intention network
        with tf.variable_scope("intention") as scope:
            self.intention_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            self.intention_state = self.get_state_variables(self.batch_size, self.intention_cell)
            if self.turn_index > 0:
                tf.get_variable_scope().reuse_variables()
            # for encoder_step_hidden_state in hidden_states:
            intention_output, intention_state = self.intention_cell(hidden_state, self.intention_state)

        # # #
        #     cT_encoder= self._concat_hidden(encoder_state)
        initial_decoder_state = []
        for i in xrange(len(intention_state)):
            b = intention_state[i]
            c = b[0]
            h = b[1]

            Dh = tf.tanh(tf.matmul(h, self.I_I))
            initial_decoder_state.append(tf.contrib.rnn.LSTMStateTuple(c, Dh))
        # #     intention_states.append(intention_hidden_state)
        #     intention_state = self.intention_state
        #     for encoder_step_hidden_state in hidden_states:
        #         intention_output, intention_state = self.intention_cell(encoder_step_hidden_state, intention_state)
        # # intention_state = self.intention_state

        # self.modified = []
        # for layer in xrange(len(encoder_state)):
        #     layer_intention_state = encoder_state[layer]
        #     layer_last_encoder_state = self.encoder_state[layer]
        #     h = layer_intention_state[1]
        #     c = layer_intention_state[0]
        #     eh = layer_last_encoder_state[1]
        #     ec = layer_last_encoder_state[0]
        #     self.kernel_i = tf.add(tf.matmul(h, self.I_I), self.intention_to_decoder_b)
        #     self.kernel_e = tf.add(tf.matmul(eh, self.I_E), self.encoder_to_intention_b)
        #     self.h_ = tf.concat([self.kernel_e, self.kernel_i], axis=1)
        #     cc = tf.concat([c, ec], axis=1)
        #     layer = tf.contrib.rnn.LSTMStateTuple(cc, self.h_)
        #     self.modified.append(layer)

        #
        with tf.variable_scope("decoder") as scope:
            if self.TRAINABLE:
                decoder_embedding_vectors = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
                for time_step in xrange(self.DECODER_NUM_STEPS):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        '''
                        plugin the intention state here!!!
                        '''
                        decoder_state = initial_decoder_state

                    # attended = decoder_state
                    # attended = self._attention(encoder_hidden_states=hidden_states, u_encoder_hidden_states=U_ah, decoder_state=decoder_state)
                    # self.e.append(e_iJ)
                    # LSTMStateTuple
                    decoder_output, decoder_state = self.decoder_cell(decoder_embedding_vectors[:, time_step, :], decoder_state)
                    self.decoder_outputs.append(decoder_output)
        #     else:
        #         gen_decoder_input = tf.constant(self.EOS, shape=(1, 1), dtype=tf.int32)
        #         for time_step in xrange(self.DECODER_NUM_STEPS):
        #             if time_step > 0:
        #                 tf.get_variable_scope().reuse_variables()
        #             else:
        #                 decoder_state = self.decoder_state
        #
        #             # attended = self._attention(encoder_hidden_states=hidden_states, u_encoder_hidden_states=U_ah,
        #             #                            decoder_state=decoder_state)
        #             gen_decoder_input_vector = tf.nn.embedding_lookup(self.embedding, gen_decoder_input)
        #             decoder_output, decoder_state = self.decoder_cell(gen_decoder_input_vector[:,0,:], decoder_state)
        #             index = self._neural_decoder_output_index(decoder_output)
        #             gen_decoder_input = tf.reshape(index,[-1, 1])
        #             self.decoder_outputs.append(decoder_output)
        #             self.internal.append(gen_decoder_input)
        #
        #     ## update states for next batch: decoder_state, intention_state
        #     # self.decoder_state_update_op = self.get_state_update_op(self.decoder_state, decoder_state)
            self.intention_state_update_op = self.get_state_update_op(self.intention_state, intention_state)
            self.encoder_state_update_op = self.get_state_update_op(self.encoder_state, decoder_state)

            ## reset op whenever a new turn begins
            # self.reset_decoder_state_op = self.get_state_reset_op(self.decoder_state, self.decoder_cell,
            #                                                       self.batch_size)
            # self.reset_encoder_state_op = self.get_state_reset_op(self.decoder_state, self.decoder_cell,
            #                                                       self.batch_size)
            # self.reset_intention_state_op = self.get_state_reset_op(self.intention_state, self.intention_cell,
            #                                                         self.batch_size)
                # logits_series = tf.matmul(decoder_output, softmax_w) + softmax_b  # Broadcasted addition
                # # logits_series = tf.nn.softmax(logits_series, dim=1)
                # y_ = labels_[:, time_step, :]

    def _attention(self, encoder_hidden_states, u_encoder_hidden_states, decoder_state):
        target_hidden_state = self._build_hidden(decoder_state)
        ## attention
        W_aS = tf.matmul(target_hidden_state, self.W_a)
        e_iJ = []
        for uj in u_encoder_hidden_states:
            WaS_UaH = tf.tanh(tf.add(W_aS, uj))
            e_ij = tf.matmul(WaS_UaH, self.v_a)  ## should be scala of batches
            e_iJ.append(e_ij)

        e_iJ = tf.stack(e_iJ)
        a_iJ = tf.reshape(tf.nn.softmax(e_iJ, dim=0), [-1, 1, self.ENCODER_NUM_STEPS])
        encoder_hidden_states = tf.reshape(encoder_hidden_states, [-1, self.ENCODER_NUM_STEPS, self.HIDDEN_UNIT])
        c_i = tf.matmul(a_iJ, encoder_hidden_states)

        attention = c_i
        attended = list()
        for b in decoder_state:
            c = b[0]
            h = b[1]
            h_ = tf.concat([h, tf.squeeze(attention, [1])], 1)
            attended_hidden_decoder_state = tf.contrib.rnn.LSTMStateTuple(c, h_)
            attended.append(attended_hidden_decoder_state)

        return attended

    def _build_hidden(self, encoder_state):
        return encoder_state[self.N_LAYER - 1][1]

    def _concat_hidden(self, encoder_state):
        states = []
        for h in encoder_state:
            states.append(h[1])
        return tf.reshape(tf.stack(states), [-1, self.N_LAYER * self.HIDDEN_UNIT])

    ## map decoder_output back to decoder_input(the index)
    ## this function is used when decoder inputs aren't given
    def _neural_decoder_output_index(self, decoder_output):
        num_classes = self.VOL_SIZE
        logits_series = tf.matmul(decoder_output, self.softmax_w) + self.softmax_b
        probs = tf.reshape(tf.nn.softmax(logits_series), [-1, 1])
        index = tf.argmax(probs)
        return index

    def _create_loss(self):
        # self.loss = tf.get_variable('loss', dtype=tf.float32, trainable=False,shape=[1])
        self.logits_ = []
        loss = 0
        for time_step in xrange(self.DECODER_NUM_STEPS):
            decoder_output = self.decoder_outputs[time_step]
            logits_series = tf.matmul(decoder_output, self.softmax_w) + self.softmax_b  # Broadcasted addition
            self.logits_.append(tf.nn.softmax(logits_series))
            y_ = self.labels_[:, time_step, :]
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_series))
            loss = loss + cross_entropy

        self.loss = loss

        # ## reset loss op
        # self.reset_loss_op = tf.assign(self.loss, [0])
        # self.plus_loss_op = tf.add(self.loss, [loss])
        # tf.summary.scalar("batch_loss", self.loss)
        self.predictions_ = [tf.argmax(logit, axis=1) for logit in self.logits_]

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def _summary(self):
        self.merged = tf.summary.merge_all()

    '''
    https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns
    '''
    def get_state_variables(self, batch_size, cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = []
        for state_c, state_h in cell.zero_state(batch_size, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)

    '''
    https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns
    '''
    def get_state_update_op(self, state_variables, new_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([tf.assign(state_variable[0], new_state[0]),
                               tf.assign(state_variable[1], new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def get_state_reset_op(self, state_variables, cell, batch_size):
        # Return an operation to set each variable in a list of LSTMStateTuples to zero
        zero_states = cell.zero_state(batch_size, tf.float32)
        return self.get_state_update_op(state_variables, zero_states)

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._summary()

def one_hot_triple(index):
    zeros_a = np.zeros(AttentionSortModel.VOL_SIZE, dtype=np.float32)
    zeros_a[index] = 1
    return zeros_a


def gen_triple(sum_ = 0):
    _input = np.random.random_integers(low=0, high=AttentionSortModel.VOL_SIZE - 2, size=AttentionSortModel.ENCODER_SEQ_LENGTH)
    # _output = np.sort(_input)

    label = []
    next_sum = 0
    ## and compute sum
    for o in _input:
        # label.append(one_hot_triple(o))
        next_sum = next_sum + o
    sum_ = sum_ % (AttentionSortModel.VOL_SIZE - 2)
    next_sum = next_sum % (AttentionSortModel.VOL_SIZE - 2)
    label.append(one_hot_triple(int(sum_)))
    label.append(one_hot_triple(int(next_sum)))
    _output = np.array([AttentionSortModel.EOS, sum_, next_sum])

    label.append(one_hot_triple(AttentionSortModel.EOS))

    return _input, _output, label, next_sum

def sort_and_sum_op_data(batch_size = AttentionSortModel.batch_size):
    batch_sum_ = np.zeros(batch_size)
    while True:
        # turn_dialog = []
        # for turn in xrange(AttentionSortModel.TURN_LENGTH):
        single_turn_encoder_inputs = []
        single_turn_decoder_inputs = []
        single_turn_labels = []
        for batch_index in xrange(batch_size):
            encoder_input, decoder_input, label, sum_ = gen_triple(sum_=batch_sum_[batch_index])
            batch_sum_[batch_index] = sum_
            single_turn_encoder_inputs.append(encoder_input)
            single_turn_decoder_inputs.append(decoder_input)
            single_turn_labels.append(label)
        stei = np.array(single_turn_encoder_inputs).reshape(batch_size, AttentionSortModel.ENCODER_NUM_STEPS)
        stdi = np.array(single_turn_decoder_inputs).reshape(batch_size, AttentionSortModel.DECODER_NUM_STEPS)
        stl = np.array(single_turn_labels)

            # turn_dialog.append([stei, stdi, stl])
        yield stei, stdi, stl

def train():

    model = AttentionSortModel()
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        gen = sort_and_sum_op_data()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../log',
                                       sess.graph)
        # _check_restore_parameters(sess, saver)
        i = 0
        all_loss = np.ones(10)
        all_loss_index = 0
        for stei, stdi, stl in gen:
            model.optimizer.run(feed_dict={model.encoder_inputs.name: stei,\
                                           model.decoder_inputs.name: stdi,\
                                           model.labels_.name: stl})

            if (i + 1) % 1 == 0:
                loss, predictions, c= sess.run(
                    [model.loss, model.predictions_, model.labels_],
                    feed_dict={model.encoder_inputs.name: stei, \
                               model.decoder_inputs.name: stdi, \
                               model.labels_.name: stl})
                all_loss_index += 1
                # all_loss[all_loss_index % 10] = loss1

                # writer.add_summary(summary, i)
                print("step and turn-1", i, stei, stdi, loss, predictions, c)
                # ki, ke, kh, dd, ii = sess.run([model.kernel_e, model.kernel_i, model.h_, model.dd, model.modified], feed_dict={model.encoder_inputs.name: stei, \
                #                model.decoder_inputs.name: stdi, \
                #                model.labels_.name: stl})
                # print(ki, ke, kh, dd, ii)
                # if np.average(all_loss) < 0.01:
                #     print('saving_model...', loss1)
                #     saver.save(sess, "../model/rnn/awi", global_step=i)

            sess.run([model.intention_state_update_op, model.encoder_state_update_op],
                     feed_dict={model.encoder_inputs.name: stei, \
                                model.decoder_inputs.name: stdi})
            i = i + 1
            model.increment_turn()

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("../model/rnn/attention"))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the SortBot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the SortBot")

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def run_sort():
    model = AttentionSortModel(False)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        while True:
            line = _get_user_input()
            # gen = sort_op_data(1)
            e_inputs = list()
            e_inputs.append(np.array([int(x) for x in line.split(",")]))
            e_inputs = np.array(e_inputs)
            predictions = np.array(np.array(sess.run(model.predictions_, feed_dict={model.encoder_inputs.name: e_inputs})))\
                .reshape([-1, AttentionSortModel.DECODER_NUM_STEPS])
            sess.run([model.encoder_state_update_op, model.decoder_state_update_op, model.intention_state_update_op],
                     feed_dict={model.encoder_inputs.name: e_inputs})
            print(e_inputs, predictions)

    return 0

if __name__ == "__main__":
    # gen = sort_and_sum_op_data()
    # for a,b,c in gen:
    #     print(a,b,c)

    train()
    # run_sort()
    # a = np.random.rand(2,2)
    # x = tf.placeholder(tf.float32, shape=(2, 2))
    # y = tf.matmul(x, x)
    # with tf.Session() as sess:
    #     print(sess.run(y, feed_dict={x:a}))