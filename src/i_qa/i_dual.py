#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model to predict the matching score between query and responses.

"""
import sys
import inspect
import random

import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf-8")

class IDual:

    batch_size = 32

    EMBEDDING_SIZE = 400
    ENCODER_SEQ_LENGTH = 5
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH =  6  ## plus 0 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH
    VOL_SIZE = 5
    # TURN_LENGTH = 3

    HIDDEN_UNIT = 256
    N_LAYER = 10

    TRAINABLE = True

    keep_prob = 0.8

    """ A retrieval-based chatbot.
    Architecture: LSTM Encoder/Encoder.
    """

    def __init__(self, trainable=True):

        self.is_training = trainable
        self.dtype = tf.float32

        self.optOp = None    # For training stage
        self.outputs = None  # For validation stage
        # build dual lstm graph
        self.build_graph()

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        # self._create_loss()
        # self._create_optimizer()
        # self._summary()

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

    def _create_placeholder(self):
        # Network input (placeholders)
        with tf.name_scope('placeholder_query'):
            self.query_seqs = tf.placeholder(tf.int32, [None, None], name='query')
            self.query_length = tf.placeholder(tf.int32, [None], name='query_length')
        #
        with tf.name_scope('placeholder_response'):
            self.response_seqs = tf.placeholder(tf.int32, [None, None], name='response')
            self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

        with tf.name_scope('placeholder_labels'):
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self.targets = tf.placeholder(tf.int32, [None], name='targets')

    def _inference(self):
        encoder_cell = self.stacked_rnn(self.HIDDEN_UNIT)

        with tf.name_scope('embedding_layer'):
            self.embedding = tf.get_variable("embedding", [self.VOL_SIZE, self.EMBEDDING_SIZE], dtype=tf.float32)
            self.embed_query = tf.nn.embedding_lookup(self.embedding, self.query_seqs)
            self.embed_response = tf.nn.embedding_lookup(self.embedding, self.response_seqs)
            if self.is_training and self.keep_prob < 1:
                self.embed_query = tf.nn.dropout(self.embed_query, keep_prob=self.keep_prob)
                self.embed_response = tf.nn.dropout(self.embed_response, keep_prob=self.keep_prob)

        self.query_output, self.query_final_state = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=self.embed_query,
            sequence_length=self.query_length,
            time_major=False,
            dtype=tf.float32)

        self.response_output, self.response_final_state = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=self.embed_response,
            sequence_length=self.response_length,
            time_major=False,
            dtype=tf.float32)

        with tf.variable_scope('bilinar_regression'):
            self.W = tf.get_variable("bilinear_W",
                                shape=[self.HIDDEN_UNIT, self.HIDDEN_UNIT],
                                initializer=tf.truncated_normal_initializer())

    def _create_loss(self):
        # use OTHER reponses in the minibatch as negative responses
        response_final_state = tf.matmul(self.response_final_state[-1].h, self.W)
        logits = tf.matmul(
            a=self.query_final_state[-1].h, b=response_final_state,
            transpose_b=True)
        self.losses = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels,
            logits=logits)
        self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")
        self.train_loss_summary = tf.summary.scalar('loss', self.mean_loss)

    def _create_optimizer(self):
        opt = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        self.optOp = opt.minimize(self.mean_loss)


    def _summary(self):
        self.training_summaries = tf.summary.merge(
            inputs=[self.train_loss_summary], name='train_monitor')

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

def step(batch, model, is_training=True):
    """ Forward/training step operation.
    """
    def zero_initial_state(batch_size, embed_dim, num_layers):
        return tuple(
            [(np.zeros((batch_size, embed_dim)),
            np.zeros((batch_size, embed_dim)))
        for _ in range(num_layers)])

    # Feed the dictionary
    feed_dict = {}
    ops = None

    feed_dict[model.query_seqs.name] = batch[0] #batch.query_seqs
    feed_dict[model.query_length.name] = batch[1] #batch.query_length
    feed_dict[model.response_seqs.name] = batch[2] #batch.response_seqs
    feed_dict[model.response_length.name] = batch[3] #batch.response_length

    if is_training:  # Training
        ops = [model.query_seqs]
        feed_dict[model.labels.name] = np.eye(len(batch[0]))
    else: # Testing or Validating
        ops = (model.outputs, model.evaluation_summaries)
        feed_dict[model.targets] = np.zeros((len(batch[0]))).astype(int)
    # Return one pass operator
    return ops, feed_dict

def generate_batch_data(batch_size = 32):
    """ Generates batches of random integer sequences,
            sequence length in [length_from, length_to],
            vocabulary in [vocab_lower, vocab_upper]
        """
    length_from = 3
    length_to = 8
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    vocab_lower = 1
    vocab_upper = IDual.VOL_SIZE
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    def pad(max_len, array):
        for i in xrange(max_len - len(array)):
            array = np.append(array, 0)
        return array

    while True:
        query_seqs = [
            [random.randint(vocab_lower, vocab_upper) for i in xrange(random_length())]
            for _ in range(batch_size)
            ]
        response_seqs = [np.sort(a) for a in query_seqs]

        query_lens = [len(a) for a in query_seqs]
        response_lens = [len(a) for a in response_seqs]

        query_seqs = [pad(length_to, a) for a in query_seqs]
        response_seqs = [pad(length_to, a) for a in response_seqs]

        yield np.array(query_seqs),\
              np.array(query_lens),\
              np.array(response_seqs),\
              np.array(response_lens)

def train():
    model = IDual(trainable=True)
    # model.build_graph()
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        gen = generate_batch_data(3)
        for query_seqs, query_lens, response_seqs, response_lens in gen:
            batch = [query_seqs, query_lens, response_seqs, response_lens]
            # opts, feed_dict = step(batch=batch, model=model, is_training=True)
            sess.run([model.embedding, feed_dict={model.query_seqs.name: batch[0], \
                                                           model.query_length.name: batch[1], \
                                                           model.response_seqs.name: batch[2],\
                                                           model.response_length.name: batch[3],
                                                           model.labels.name: np.eye(len(batch[0]))})

def main(mode='train'):
    if mode == 'train':
        train()
    else:
        pass

if __name__ == '__main__':
    main()

