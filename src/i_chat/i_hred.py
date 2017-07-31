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
from tensorflow.python.layers.core import Dense

# from seq2seq.encoders import rnn_encoder
# from seq2seq.decoders import basic_decoder

from i_config import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class AttentionSortModel:

    batch_size = 32

    EMBEDDING_SIZE = 400
    ENCODER_SEQ_LENGTH = 5
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH = 6  # plus 0 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH
    # TURN_LENGTH = 3

    HIDDEN_UNIT = 256
    N_LAYER = 10

    TRAINABLE = True

    def __init__(self, data_config, trainable=True):
        print('initializing model...')
        self.TRAINABLE = trainable
        self.turn_index = 0
        self.data_config = data_config

        self.VOL_SIZE = self.data_config.VOL_SIZE
        self.EOS = self.data_config.EOS

        if trainable:
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        else:
            self.mode = tf.contrib.learn.ModeKeys.INFER

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
        self.labels_ = tf.placeholder(
            tf.int32, shape=(None, None), name='labels_')
        with tf.variable_scope("encoder") as scope:
            self.encoder_inputs = tf.placeholder(
                tf.int32, shape=(None, None), name="encoder_inputs")
            self.encoder_inputs_length = tf.placeholder(
                tf.int32, shape=(None,), name="encoder_inputs_length")
        with tf.variable_scope("decoder") as scope:
            self.decoder_inputs = tf.placeholder(
                tf.int32, shape=(None, None), name="decoder_inputs")
            self.decoder_inputs_length = tf.placeholder(
                tf.int32, shape=(None,), name="decoder_inputs_length")
        # self.mask = tf.placeholder(tf.float32, shape=(None, self.DECODER_SEQ_LENGTH), name="mask")

    def init_state(self, cell, batch_size):
        if self.TRAINABLE:
            return cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        else:
            return cell.zero_state(batch_size=1, dtype=tf.float32)

    def variable(self, shape, name):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name=name)

    # refer HEAVILY to the paper:  https://arxiv.org/pdf/1409.0473.pdf supplementary part
    # remember! by default RNN state DOES NOT keep in the next batch!!!!!!
    def _inference(self):

        self.embedding = tf.get_variable(
            "embedding", [self.VOL_SIZE, self.EMBEDDING_SIZE], dtype=tf.float32)
        num_classes = self.VOL_SIZE
        # use softmax to map decoder_output to number(0-5,EOS)
        self.softmax_w = self.variable(
            name="softmax_w", shape=[self.HIDDEN_UNIT, num_classes])
        self.softmax_b = self.variable(name="softmax_b", shape=[num_classes])

        # prepare to compute c_i = \sum a_{ij}h_j, encoder_states are h_js
        hidden_states = []
        self.W_a = self.variable(name="attention_w_a", shape=[
                                 self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.U_a = self.variable(name="attention_u_a", shape=[
                                 self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.v_a = self.variable(name="attention_v_a", shape=[
                                 1, self.EMBEDDING_SIZE])

        # connect intention with decoder
        # connect intention with intention
        self.I_E = self.variable(name="intention_e", shape=[
                                 self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.encoder_to_intention_b = self.variable(
            name="encoder_intention_b", shape=[self.HIDDEN_UNIT])
        self.I_I = self.variable(name="intention_i", shape=[
                                 self.HIDDEN_UNIT, self.HIDDEN_UNIT])
        self.intention_to_decoder_b = self.variable(
            name="intention_decoder_b", shape=[self.HIDDEN_UNIT])
        # self.C = self.variable(name="attention_C", shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT])

        # encoder_params = rnn_encoder.StackBidirectionalRNNEncoder.default_params()
        # encoder_params["rnn_cell"]["cell_params"][
        #     "num_units"] = self.HIDDEN_UNIT
        # encoder_params["rnn_cell"]["cell_class"] = "BasicLSTMCell"
        # encoder_params["rnn_cell"]["num_layers"] = self.N_LAYER

        with tf.variable_scope("encoder") as scope:
            encoder_embedding_vectors = tf.nn.embedding_lookup(
                self.embedding, self.encoder_inputs)
            encoder_fw_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            encoder_bw_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            self.encoder_initial_fw_state = self.get_state_variables(
                self.batch_size, encoder_fw_cell)
            self.encoder_initial_bw_state = self.get_state_variables(
                self.batch_size, encoder_bw_cell)
            ((outputs_fw, outputs_bw), (state_fw, state_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell, cell_bw=encoder_bw_cell,
                                                inputs=encoder_embedding_vectors,
                                                sequence_length=self.encoder_inputs_length,
                                                initial_state_fw=self.encoder_initial_fw_state,
                                                initial_state_bw=self.encoder_initial_bw_state,
                                                dtype=tf.float32)
        encoder_final_state_c = tf.concat(
            (state_fw[self.N_LAYER - 1][0],
             state_bw[self.N_LAYER - 1][0]),
            1)

        encoder_final_state_h = tf.concat(
            (state_fw[self.N_LAYER - 1][1],
             state_bw[self.N_LAYER - 1][1]),
            1)

        encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        hidden_state = tf.reshape(
            encoder_final_state[1], shape=(-1, self.HIDDEN_UNIT * 2))

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
            self.decoder_state = self.get_state_variables(
                self.batch_size, self.decoder_cell)
        #
        # building intention network
        with tf.variable_scope("intention") as scope:
            self.intention_cell = self.stacked_rnn(self.HIDDEN_UNIT)
            self.intention_state = self.get_state_variables(
                self.batch_size, self.intention_cell)
            if self.turn_index > 0:
                tf.get_variable_scope().reuse_variables()
            # for encoder_step_hidden_state in hidden_states:
            intention_output, intention_state = self.intention_cell(
                hidden_state, self.intention_state)

        # # #
        #     cT_encoder= self._concat_hidden(encoder_state)
        initial_decoder_state = []
        for i in xrange(len(intention_state)):
            b = intention_state[i]
            c = b[0]
            h = b[1]

            Dh = tf.tanh(tf.matmul(h, self.I_I))
            initial_decoder_state.append(tf.contrib.rnn.LSTMStateTuple(c, Dh))
        # print(len(initial_decoder_state))
        initial_decoder_state = tuple(initial_decoder_state)
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

        # *****************************************mark************************************************************
        # with tf.variable_scope("decoder") as scope:
        #     if self.TRAINABLE:
        #         decoder_embedding_vectors = tf.nn.embedding_lookup(
        #             self.embedding, self.decoder_inputs)
        #         self.decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=self.decoder_cell,
        #                                                                 inputs=decoder_embedding_vectors,
        #                                                                 sequence_length=self.decoder_inputs_length,
        #                                                                 initial_state=initial_decoder_state,
        #                                                                 dtype=tf.float32
        #                                                                 )
        #         self.intention_state_update_op = self.get_state_update_op(
        #             self.intention_state, intention_state)
        #         self.encoder_state_update_op = self.get_state_update_op(
        #             self.encoder_initial_fw_state, decoder_state)

        # *****************************************mark end********************

        # ***************try another way to decode*********************

        with tf.variable_scope("decoder") as scope:
            if self.TRAINABLE:
                decoder_embedding_vectors = tf.nn.embedding_lookup(
                    self.embedding, self.decoder_inputs)
                output_layer = Dense(self.VOL_SIZE,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.max_target_sequence_length = tf.reduce_max(
                    self.decoder_inputs_length, name='max_target_len')

                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedding_vectors,
                                                                    sequence_length=self.decoder_inputs_length,
                                                                    time_major=False)

                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                   helper=training_helper,
                                                                   initial_state=initial_decoder_state,
                                                                   output_layer=output_layer)

                self.decoder_output, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                          impute_finished=True,
                                                                                          maximum_iterations=self.max_target_sequence_length)
                self.intention_state_update_op = self.get_state_update_op(
                    self.intention_state, intention_state)
                self.encoder_state_update_op = self.get_state_update_op(
                    self.encoder_initial_fw_state, decoder_state)

    def _attention(self, encoder_hidden_states, u_encoder_hidden_states, decoder_state):
        target_hidden_state = self._build_hidden(decoder_state)
        # attention
        W_aS = tf.matmul(target_hidden_state, self.W_a)
        e_iJ = []
        for uj in u_encoder_hidden_states:
            WaS_UaH = tf.tanh(tf.add(W_aS, uj))
            e_ij = tf.matmul(WaS_UaH, self.v_a)  # should be scala of batches
            e_iJ.append(e_ij)

        e_iJ = tf.stack(e_iJ)
        a_iJ = tf.reshape(tf.nn.softmax(e_iJ, dim=0),
                          [-1, 1, self.ENCODER_NUM_STEPS])
        encoder_hidden_states = tf.transpose(encoder_hidden_states, [1, 0, 2])
        c_i = tf.matmul(a_iJ, encoder_hidden_states)

        attention = c_i
        attended = list()
        for b in decoder_state:
            c = b[0]
            h = b[1]
            h_ = tf.concat([h, tf.squeeze(attention, [1])], 1)
            attended_hidden_decoder_state = tf.contrib.rnn.LSTMStateTuple(
                c, h_)
            attended.append(attended_hidden_decoder_state)

        return attended

    def _build_hidden(self, encoder_state):
        return encoder_state[self.N_LAYER - 1][1]

    def _concat_hidden(self, encoder_state):
        states = []
        for h in encoder_state:
            states.append(h[1])
        return tf.reshape(tf.stack(states), [-1, self.N_LAYER * self.HIDDEN_UNIT])

    # map decoder_output back to decoder_input(the index)
    # this function is used when decoder inputs aren't given
    def _neural_decoder_output_index(self, decoder_output):
        num_classes = self.VOL_SIZE
        logits_series = tf.matmul(
            decoder_output, self.softmax_w) + self.softmax_b
        probs = tf.reshape(tf.nn.softmax(logits_series), [-1, 1])
        index = tf.argmax(probs)
        return index

    def _create_loss(self):
        self.training_logits_ = tf.identity(
            self.decoder_output.rnn_output, 'logits')
        masks = tf.sequence_mask(
            self.decoder_inputs_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.training_logits_,
            self.labels_,
            masks)
        self.predictions_ = tf.argmax(self.training_logits_, axis=2)

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

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
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial
        # state
        return tuple(state_variables)

    '''
    https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns
    '''

    def get_state_update_op(self, state_variables, new_states):
        # Add an operation to update the train states with the last state
        # tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([tf.assign(state_variable[0], new_state[0]),
                               tf.assign(state_variable[1], new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def get_state_reset_op(self, state_variables, cell, batch_size):
        # Return an operation to set each variable in a list of LSTMStateTuples
        # to zero
        zero_states = cell.zero_state(batch_size, tf.float32)
        return self.get_state_update_op(state_variables, zero_states)

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._summary()


def create_mask():
    mask = np.ones(shape=(AttentionSortModel.batch_size,
                          AttentionSortModel.DECODER_SEQ_LENGTH), dtype=np.int32)
    return np.array(mask)


def train():
    config = Config('../../data/classified/qa/qa.txt')
    # config = Config('../../data/small_poem.txt')
    model = AttentionSortModel(data_config=config, trainable=True)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        gen = config.generate_batch_data(AttentionSortModel.batch_size)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../log',
                                       sess.graph)
        # _check_restore_parameters(sess, saver)
        i = 0
        all_loss = np.ones(10)
        all_loss_index = 0
        max_loss = 0.01
        mask = create_mask()
        for enci, deci, lab, encil, decil in gen:
            # print(enci)
            # print(deci)
            # print(lab)
            model.optimizer.run(feed_dict={model.encoder_inputs.name: enci,
                                           model.encoder_inputs_length.name: encil,
                                           model.decoder_inputs.name: deci,
                                           model.decoder_inputs_length.name: decil,
                                           model.labels_.name: lab})

            if (i + 1) % 1 == 0:
                loss, predictions, logits, c = sess.run(
                    [model.loss, model.predictions_,
                        model.training_logits_, model.labels_],
                    feed_dict={model.encoder_inputs.name: enci,
                               model.encoder_inputs_length.name: encil,
                               model.decoder_inputs.name: deci,
                               model.decoder_inputs_length.name: decil,
                               model.labels_.name: lab})
                all_loss_index += 1
                # all_loss[all_loss_index % 10] = loss1

                # writer.add_summary(summary, i)
                # if loss < 0.3:
                print("train_logits shape:", logits.shape)
                print("predictions shape:", predictions.shape)
                print("step and turn-1", i, config.recover(enci[0]), config.recover(
                    deci[0]), loss, config.recover(predictions[0]), c[0])
                # ki, ke, kh, dd, ii = sess.run([model.kernel_e, model.kernel_i, model.h_, model.dd, model.modified], feed_dict={model.encoder_inputs.name: stei, \
                #                model.decoder_inputs.name: stdi, \
                #                model.labels_.name: stl})
                # print(ki, ke, kh, dd, ii)
                if loss < max_loss:
                    max_loss = loss * 0.7
                    print('saving model...', i, loss)
                    saver.save(sess, "../../model/qa/i_hred", global_step=i)
                if i % 1000 == 0:
                    print('safe_mode saving model...', i, loss)
                    saver.save(sess, "../../model/qa/i_hred", global_step=i)

            sess.run([model.intention_state_update_op, model.encoder_state_update_op],
                     feed_dict={model.encoder_inputs.name: enci,
                                model.encoder_inputs_length.name: encil,
                                model.decoder_inputs.name: deci,
                                model.decoder_inputs_length.name: decil,
                                model.labels_.name: lab})
            i = i + 1
            model.increment_turn()


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname("../../model/qa/i_hred"))
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
