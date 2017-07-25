#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf-8")

# import tensorflow.contrib.seq2seq.Helper
from seq2seq.encoders import rnn_encoder
from seq2seq.decoders import basic_decoder

from seq2seq.contrib.seq2seq import decoder as contrib_decoder

PAD = 0
EOS = 1


vocab_size = 10
input_embedding_size = 50

# 第一层的encoder RNN cell 的 hidden_state_size
encoder1_hidden_units = 50
# 第二层的encoder RNN cell 的 hidden_state_size
# 因为要记忆相对大量的context，所以 “*2”
encoder2_hidden_units = encoder1_hidden_units * 2

# decoder 的 hidden_state_size
# encoder2 使用 unidirectional_rnn
# 注意 encoder2 不能使用 bidirectional_rnn
decoder_hidden_units = encoder2_hidden_units

import helpers as data_helpers
batch_size = 4
round_num = 3

# 一个generator，每次产生一个minibatch的随机样本

batches = data_helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size*round_num)

def demo_mult_rounds(batches, batch_size, round_num):
    data = next(batches)
    mb = list()
    id = 0
    for i in range(batch_size):
        mb.append([])
        for j in range(round_num):
            mb[-1].append(data[id])
            id += 1
    return mb

print('产生%d组的sequences, \n'
      '每一组sequence包含%d句长度不一（最短3，最长8）的sequence, \n'
      '其中前十组是:\n' % (batch_size, round_num))

for seq in demo_mult_rounds(batches, batch_size, round_num):
    print('%s\n' % seq)

tf.reset_default_graph()
sess = tf.InteractiveSession()
mode = tf.contrib.learn.ModeKeys.TRAIN

with tf.name_scope('minibatch_encoder1'):
    # 一个 minibatch 包含 batch_size * round_num 个 sequences
    encoder1_inputs = tf.placeholder(shape=(batch_size * round_num, None),
                                     dtype=tf.int32,
                                     name='encoder1_inputs')
    encoder1_inputs_length = tf.placeholder(shape=(batch_size * round_num,),
                                            dtype=tf.int32,
                                            name='encoder1_inputs_length')

with tf.name_scope('minibatch_encoder2'):
    encoder2_inputs_length = tf.placeholder(shape=(batch_size,),
                                            dtype=tf.int32,
                                            name='encoder2_inputs_length')

with tf.name_scope('minibatch_decoder'):
    decoder_targets = tf.placeholder(shape=(batch_size * round_num, None),
                                     dtype=tf.int32,
                                     name='decoder_targets')

    decoder_inputs = tf.placeholder(shape=(batch_size * round_num, None),
                                    dtype=tf.int32,
                                    name='decoder_inputs')
    decoder_inputs_length = tf.placeholder(shape=(batch_size * round_num,),
                                           dtype=tf.int32,
                                           name='decoder_inputs_length')
# 每个句子encoding的超参数
encoder1_params = rnn_encoder.StackBidirectionalRNNEncoder.default_params()
encoder1_params["rnn_cell"]["cell_params"]["num_units"] = encoder1_hidden_units
encoder1_params["rnn_cell"]["cell_class"] = "BasicLSTMCell"
encoder1_params["rnn_cell"]["num_layers"] = 2

# 第一层 embedding
with tf.name_scope('embedding'):
    input_embeddings = tf.Variable(
        tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
        dtype=tf.float32)

with tf.name_scope('ENC-level1'):

    encoder1_inputs_embedded = tf.nn.embedding_lookup(
        input_embeddings, encoder1_inputs)
    encode_fn1 = rnn_encoder.StackBidirectionalRNNEncoder(
        encoder1_params, mode)
    encoder1_output = encode_fn1(
        encoder1_inputs_embedded, encoder1_inputs_length)

print('outputs: %s\n\n' % repr(encoder1_output.outputs))
print('final state: %s\n\n' % repr(encoder1_output.final_state))

with tf.name_scope('level1-states'):
    encoder1_final_state_c = tf.concat(
        (encoder1_output.final_state[0][1].c,
         encoder1_output.final_state[1][1].c),
        1)

    encoder1_final_state_h = tf.concat(
        (encoder1_output.final_state[0][1].h,
         encoder1_output.final_state[1][1].h),
        1)

    encoder1_final_state = tf.nn.rnn_cell.LSTMStateTuple(
        c=encoder1_final_state_c,
        h=encoder1_final_state_h
    )

# 将context做encoding的超参数
encoder2_params = rnn_encoder.UnidirectionalRNNEncoder.default_params()
encoder2_params["rnn_cell"]["cell_params"]["num_units"] = encoder2_hidden_units
encoder2_params["rnn_cell"]["cell_class"] = "BasicLSTMCell"
encoder2_params["rnn_cell"]["num_layers"] = 2

# 第二层 embedding
print(repr(encoder1_final_state))

with tf.name_scope('ENC-level2'):
    # 1. reshape from (batch_size x round_num, hidden_state_size)
    #   to (batch_size, round_num, hidden_state_size)
    encoder2_inputs = tf.reshape(encoder1_final_state.h,
                                 [-1, round_num, encoder2_hidden_units])

    # 2. 共batch_size个样本，每个样本长度为 round_num, 每个元素是一个原始样本的rnn_encoder_final_state
    encode_fn2 = rnn_encoder.UnidirectionalRNNEncoder(
        encoder2_params, mode)
    encoder2_output = encode_fn2(encoder2_inputs, encoder2_inputs_length)

# 3. 将batch_size个样本的各个round_num个元素的 output 作为 decoding context
with tf.name_scope('level2_outputs'):
    context_state = tf.reshape(
        encoder2_output.outputs,
        [batch_size * round_num, encoder2_hidden_units])

# 准备新的输入

with tf.name_scope('decoder_input'):
    decoder_inputs_embedded = tf.nn.embedding_lookup(
        input_embeddings, decoder_inputs)

    context_state = tf.tile(context_state,
                            [1, tf.shape(decoder_inputs_embedded)[1]])
    context_state = tf.reshape(context_state,
                               [batch_size * round_num,
                                tf.shape(decoder_inputs_embedded)[1],
                                encoder2_hidden_units])

    decoder_inputs_embedded = tf.concat(
        (decoder_inputs_embedded,
         context_state
         ), axis=-1
    )

from seq2seq.contrib.seq2seq import helper as decode_helper
with tf.name_scope('decoder_helper'):
    helper_ = decode_helper.TrainingHelper(
        inputs = decoder_inputs_embedded,
        sequence_length = decoder_inputs_length)

decode_params = basic_decoder.BasicDecoder.default_params()
decode_params["rnn_cell"]["cell_params"]["num_units"] = decoder_hidden_units
decode_params["max_decode_length"] = batch_size * round_num + 5

with tf.name_scope('decoder'):
    decoder_fn = basic_decoder.BasicDecoder(params=decode_params,
                                            mode=mode,
                                            vocab_size=vocab_size)
    decoder_output, decoder_state = decoder_fn(
        encoder1_final_state,
        helper_)

with tf.name_scope('loss'):
    indices = tf.constant(
        [[x] for x in range(round_num-1, batch_size*round_num, round_num)],
        dtype=tf.int32)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(tf.gather_nd(params = decoder_targets,
                                           indices = indices),
                              depth=vocab_size, dtype=tf.float32),
            logits=tf.gather_nd(params = tf.transpose(decoder_output.logits,
                                             perm = [1, 0, 2]),
                               indices = indices)
        )
    )

    train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

import os

log_path = os.path.join(os.getcwd(), 'arch-hred-basic')
summary_writer = tf.summary.FileWriter(log_path, sess.graph)


# 保存模型
# word2vec参数的单词和词向量部分分别保存到了metadata和ckpt文件里面
saver = tf.train.Saver()
saver.save(sess, os.path.join(log_path, "model.ckpt"))

batch = next(batches)

cumbatch = []
for i in range(len(batch)):
    if i%round_num==0:
        cumbatch.append(batch[i])
    else:
        cumbatch.append(batch[i] + cumbatch[-1])

encoder_inputs_, encoder_inputs_length_ = data_helpers.batch(batch)
decoder_targets_, _ = data_helpers.batch(
    [(sequence) + [EOS] for sequence in cumbatch]
)
decoder_inputs_, decoder_inputs_length_ = data_helpers.batch(
    [[EOS] + (sequence) for sequence in cumbatch]
)

def next_feed():
    batch = next(batches)

    cumbatch = []
    for i in range(len(batch)):
        if i%round_num==0:
            cumbatch.append(batch[i])
        else:
            cumbatch.append(batch[i] + cumbatch[-1])

    encoder_inputs_, encoder1_inputs_length_ = data_helpers.batch(batch)
    print(encoder_inputs_, encoder1_inputs_length_)
    encoder2_inputs_length_ = np.array([round_num]*batch_size)
    decoder_targets_, _ = data_helpers.batch(
        [(sequence) + [EOS] for sequence in cumbatch]
    )
    decoder_inputs_, decoder_inputs_length_ = data_helpers.batch(
        [[EOS] + (sequence) for sequence in cumbatch]
    )
    # 在feedDict里面，key可以是一个Tensor
    return {
        encoder1_inputs: encoder_inputs_.T,
        decoder_inputs: decoder_inputs_.T,
        decoder_targets: decoder_targets_.T,
        encoder1_inputs_length: encoder1_inputs_length_,
        encoder2_inputs_length: encoder2_inputs_length_,
        decoder_inputs_length: decoder_inputs_length_
    }


loss_track = []
max_batches = 3001
batches_in_epoch = 100

try:
    # 一个epoch的learning
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_output.predicted_ids, fd)
            for i, (inp, targ, pred) in enumerate(
                    zip(fd[encoder1_inputs],
                        fd[decoder_targets],
                        predict_.T)):
                if i in [0, round_num - 1]:
                    print('  sample {}:'.format(i + 1))
                    print('    targets     > {}'.format(targ))
                    print('    predicted > {}'.format(pred))
                if i == round_num - 1:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')

import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

