

import os
import traceback
import time
import numpy as np
from datetime import datetime
from six.moves import xrange

import tensorflow as tf
import data_input
import model
import evaluation
import utils


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('initial_learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'Epochs after which learning rate decays.')
flags.DEFINE_float('learning_rate_decay_factor', 0.01, 'Learning rate decay factor.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')

flags.DEFINE_string('log_dir', 'log/mnist/', 'Directory to put the log data. Set this -logdir in tensorboard')
flags.DEFINE_string('img_dir', 'data/mnist/', 'Directory of images.')
flags.DEFINE_integer('test_size', 10000, 'Size of testing data. Rest will be used for training.')
flags.DEFINE_integer('image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('image_height', 28, 'x, y size of image.')
flags.DEFINE_boolean('is_jpeg', False, 'jpeg = True, png = False')   

flags.DEFINE_integer('max_steps', 100000, 'Max. number of steps to run trainer.')
flags.DEFINE_integer('num_epochs', 1000, 'Max. number of epochs to run trainer.')

flags.DEFINE_integer('num_classes', 10, 'Number of classes to predict')   


def predict():
    if not tf.gfile.Exists(FLAGS.img_dir):
        print("Folder {0} does not exist".format(FLAGS.img_dir))
        return

    if not tf.gfile.Exists(FLAGS.log_dir):
        print("Folder {0} does not exist".format(FLAGS.log_dir))
        return

    with tf.Graph().as_default():
        data_sets = data_input.read_labeled_image_batches(FLAGS)

        images_placeholder, labels_placeholder = utils.create_placeholder_inputs(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width)
        logits = model.inference(images_placeholder, FLAGS.batch_size, FLAGS.num_classes)

        correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            sess.run(init)
            
            try:
                # Start the queue runners.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Restore variables from disk.
                saver.restore(sess, FLAGS.log_dir + "model.ckpt-10001")
                print("Model restored.")
                
                print("Evaluate")
                evaluation.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)
            finally:
                print("\nWaiting for all threads...")
                coord.request_stop()
                coord.join(threads)



# http://www.samansari.info/2016/02/learning-tensorflow-2-training.html
# https://niektemme.com/2016/02/21/tensorflow-handwriting/
def main(argv=None):
    try:
        predict()

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")


if __name__ == '__main__':
    tf.app.run()