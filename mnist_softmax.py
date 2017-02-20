# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

display_step = 1
logs_path = '/tmp/tensorflow_logs/softmax'

total_epoch = 200
batch_size = 100
total_example = 50000
total_batch = 500



def main(_):


  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]), name="Weights")
  b = tf.Variable(tf.zeros([10]), name="Bias")

  with tf.name_scope('Model'):
    y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  with tf.name_scope('SGD'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  with tf.name_scope('Accuracy'):
      acc = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
      acc = tf.reduce_mean(tf.cast(acc, tf.float32))

  tf.summary.scalar("loss", cross_entropy)
  tf.summary.scalar("accuracy", acc)


  merged_summary_op = tf.summary.merge_all()


  with tf.Session() as sess:
    avg_cost = 0.
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(total_epoch):
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x: batch_xs, y_: batch_ys})
        summary_writer.add_summary(summary, epoch * (total_batch)+i)
        avg_cost += c/total_batch
      if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    print("Accuracy:", sess.run(acc, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)