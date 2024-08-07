''' Build a 2-layer fully connected neural network'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
    '''Build the moddel up to where it may be used for inference
        Args:
            images: Image placeholder (input data)
            images_pixels: Number of pixel per image.
            hidden_units: Size of the first (hidden) layer.
            classes: Number of possible image classes/labels
            reg_constant: Regularization constant (default 0).

        Returns:
            logits: Output tensor containing the compute logits
    '''

    #Layer 1
    with tf.variable_scope('Layer1'):
        weights = tf.get_variable(
            name='weights',
            shape=[image_pixels, hidden_units],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(image_pixels)),
            regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
            )
        )

        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')

        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

    #Layer 2
    with tf.variable_scope('Layer 2'):
        weights = tf.get_variable('weights', [hidden_units, classes],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(hidden_units))),
            regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
            )
        
        biases = tf.Variable(tf.zeros([classes]), name="biases")

        logits = tf.matmul(hidden, weights) + biases

        tf.summary.histogram('logits', logits)

        return logits
    
def loss(logits, labels):
    '''Calculate the loss from logits and labels

    Args:
        logits: logits tensor, float - [batch size, number of classes].
        labels: labels tensor, int64 - [batch size].

    Return:
        loss: Loss tensor of type float
    '''

    with tf.name_scope('Loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy'
            )
        )

        loss = cross_entropy + tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES
        ))

        tf.summary.scalar('loss', loss)

        return loss

def training(loss, learning_rate):
    '''Sets up the training operation.
    
    Creates an optimizer and applies the gradients to all trainable variables.

    Args:
        loss: Loss tender, from loss().
        learning_rate: the learning rate to use for gradient descent

    Returns:
        traing_step: the op for trainings

    '''

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step

def evaluation(logits, labels):
    '''Evaluates the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch size, number of classes]
        labels: Labels tensor, int64 - [batch size]

    Returns:
        accuracy: the percentage of images where the class was correctly predicted

    '''

    with tf.name_scope('Accuracy'):

        correct_prediction = tf.equal(tf.argmax(logits,1), labels)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('train_accuracy', accuracy)

        return accuracy

    