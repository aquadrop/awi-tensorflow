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

Demonstrate that we can build model in a seperate class.
And load and save the model using saver

The model learns a simple linear regression
"""
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import inspect

class SimpleModel:
    def __init__(self):
        print("Hello Loading/Saving Model...")

    def _create_placeholder(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1), name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1), name="y")

    def _inference(self):
        self.w = tf.Variable(tf.zeros([1, 1]), name="weights")
        self.b = tf.Variable(tf.zeros([1]), name='biases')

        self.linear_y = tf.matmul(self.x, self.w) + self.b

    def _create_optimizer(self):
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.linear_y)))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_optimizer()

batch_size = 5
def generate_data():
    while True:
        inputs = np.random.rand(batch_size)
        outputs = []
        for i in inputs:
            output = 2 * i + 1.0
            outputs.append(output)
        yield inputs.reshape(batch_size, 1), np.asarray(outputs).reshape(batch_size, 1)

def main():
    model = SimpleModel()
    model.build_graph()
    data = generate_data()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for inputs, outputs in data:
            model.optimizer.run(feed_dict={model.x.name: inputs, model.y.name: outputs})
            saver.save(sess, "../model/model")
            print(sess.run(model.loss, feed_dict={model.x.name: inputs, model.y.name: outputs}))

def use():
    model = SimpleModel()
    model.build_graph()
    saver = tf.train.Saver()


if __name__ == "__main__":
    main()
