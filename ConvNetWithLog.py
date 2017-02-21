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

total_epoch = 20
batch_size = 50
logs_path = '/tmp/tensorflow_logs/convNet1'
display_step = 1



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="Weight")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="Bias")
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    total_sample = len(mnist.train.images)
    batch_per_epoch = int(total_sample/batch_size)
    print("%d training samples and %d batch per epoch"%(total_sample, batch_per_epoch))

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])


    #First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    #Second Convolutional Layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # Densely Connected Layer

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout to avoid overfiting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer, softmax regression
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train and evaluate,
    # 1.We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
    # 2.We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
    # 3.We will add logging to every 100th iteration in the training process.
    with tf.name_scope('Loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    with tf.name_scope('ADAM'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        avg_cost = 0.

        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for epoch in range(total_epoch):
            for i in range(batch_per_epoch):
                batch = mnist.train.next_batch(batch_size)
                _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                summary_writer.add_summary(summary, epoch*batch_per_epoch+i)
                avg_cost += c/batch_per_epoch
                # print("batch %d with loss %g" % (i+epoch*batch_per_epoch, c))
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("test accuracy %g" % sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    # Line below is important, otherwise you will get exception 
    # when run it in ipython related with placeholder variable, 
    # because session data is not fully cleaned up.
    tf.reset_default_graph()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    sys.exit()
