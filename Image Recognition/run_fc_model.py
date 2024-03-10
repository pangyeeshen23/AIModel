# Train and evaluates a fully-connected nueral net classifier for CIFAR-10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import two_layer_fc

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400, 'Batch size. Must divide dataset sized without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs', 'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

data_sets = data_helpers.load_data()

images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='images')
label_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS, FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

loss = two_layer_fc.loss(logits, label_placeholder)

train_step = two_layer_fc.training(loss, FLAGS.learning_rate)

accuracy = two_layer_fc.evaluation(logits, label_placeholder)

summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
    batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size, FLAGS.max_steps)

    for i in range(FLAGS.max_steps):
        
        batch = next(batches)
        image_batch, label_batch = zip(*batch)

        feed_dict = {
            images_placeholder: image_batch,
            label_placeholder: label_batch
        }

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict) 
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)

        sess.run([train_step, loss], feed_dict=feed_dict)

        if(i + 1) % 1000 == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            print("Saved Checkpoint")

        test_accuracy = sess.run(accuracy, feed_dict={
            images_placeholder: data_sets['images_test'],
            label_placeholder: data_sets['labels_test']
        })
        print('Test accuracy {:g}'.format(test_accuracy))

        endTime = time.time()
        print("Total time: {:5.2f}s".format(endTime - beginTime))