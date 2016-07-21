
#
# Convolutional Neural Networks with Tensorflow
# Peer David - 2016
#
# data
#  car
#   0 = oldtimer
#   1 = super
#   2 = estate
#
#  mnist
#   0 = 0
#    ...
#   9 = 9
#
#  object_categories
#   0 = faces
#   1 = airplane
#   2 = motorbike
#
# References
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
#   https://www.tensorflow.org/versions/r0.9/how_tos/variable_scope/index.html 
#   https://github.com/HamedMP/ImageFlow/blob/master/example_project/my_cifar_train.py
#
# ToDo:
# - Refactoring (Training, Evaluation, Visualization, Input, Model)
# - Get number of classes by number of folders of file system
# - Evaluation -> https://www.tensorflow.org/versions/r0.9/how_tos/reading_data/index.html
#      - The training process reads training input data and periodically writes checkpoint files with all the trained variables.
#      - The evaluation process restores the checkpoint files into an inference model that reads validation input data.
# - classify.py
# - Dropout
# - ... and somewhere inside "def train():" after calling "inference()" visualize conv1 layer = https://gist.github.com/kukuruza/03731dc494603ceab0c5
# - Asser loss nan
# 

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


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

#
# Hyperparameters
#
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('initial_learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'Epochs after which learning rate decays.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('test_size', 10000, 'Size of testing data. Rest will be used for training.')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_string('log_dir', 'log_dir/current/', 'Directory to put the log data.')
flags.DEFINE_string('img_dir', 'data/mnist/', 'Directory of images.')
flags.DEFINE_integer('orig_image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('orig_image_height', 28, 'x, y size of image.')
flags.DEFINE_integer('image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('image_height', 28, 'x, y size of image.')
flags.DEFINE_integer('image_pixels', 28 * 28, 'num of pixels per image.')
flags.DEFINE_integer('num_classes', 10, 'Number of image classes')  
flags.DEFINE_boolean('is_jpeg', False, 'jpeg = True, png = False')   
  
      
#
# Helper functions
#          
def _create_placeholder_inputs(batch_size, image_height, image_width):
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))
    labels_pl = tf.placeholder(tf.int32, shape=batch_size)
    return images_pl, labels_pl


def _create_train_loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='loss_total')
    
    
def _create_train_op(total_loss, global_step, num_images_per_epoch_of_train):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = num_images_per_epoch_of_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
        
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        ret = tf.no_op(name='train')

    return ret

  
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
    
        
#
# M A I N
#
if __name__ == '__main__':
    try:
        
        # Create log dir if not exists
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            data_sets = data_input.read_labeled_image_batches(FLAGS)
            train_data_set = data_sets.train
            test_data_set = data_sets.test
          
            images_placeholder, labels_placeholder = _create_placeholder_inputs(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width)
                    
            # Build a Graph that computes predictions from the inference model.
            # We use the same weight's etc. for the training and testing
            logits = model.inference(images_placeholder, FLAGS)
                            
            # Accuracy
            correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
            eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

            # Add to the Graph the Ops for loss calculation.
            train_loss = _create_train_loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            global_step = tf.Variable(0, trainable=False)
            train_op = _create_train_op(train_loss, global_step, train_data_set.size)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.all_variables())

            # Add tensorboard summaries
            tf.image_summary('image_train', train_data_set.images, max_images = 5)
            tf.image_summary('image_test', test_data_set.images, max_images = 5)
  
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            
            # Add the variable initializer Op.
            init = tf.initialize_all_variables()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # And then after everything is built:
            # Run the Op to initialize the variables.
            sess.run(init)
    
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
           
            try:
                # Start the training loop.
                for step in xrange(FLAGS.max_steps):
                    if coord.should_stop():
                        break
                        
                    start_time = time.time()
                    
                    train_feed = utils.create_feed_data(sess, images_placeholder, labels_placeholder, train_data_set)
                    _, loss_value = sess.run([train_op, train_loss], feed_dict=train_feed)
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    duration = time.time() - start_time

                    # Print step loss etc.
                    if step % 10 == 0:
                        num_examples_per_step = train_data_set.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        print ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                                    'sec/batch)' % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
                    
                    # Write summary
                    if step % 100 == 0:
                        summary_str = sess.run([summary_op], feed_dict=train_feed)
                        summary_writer.add_summary(summary_str[0], step)
                        
                    # Calculate accuracy      
                    if step % 500 == 0 and step != 0:                                 
                        precision = evaluation.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, test_data_set)
                        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_test", simple_value=precision)])
                        summary_writer.add_summary(summary, step) 
                        
                        precision = evaluation.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, train_data_set)
                        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_train", simple_value=precision)])
                        summary_writer.add_summary(summary, step) 

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step)
            
            
            except tf.errors.OutOfRangeError:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            
            finally:
                # Finished
                print("\nWaiting for all threads...")
                coord.request_stop()
                coord.join(threads)
                print("Closing session...\n")
                sess.close()
 
    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")