
#
# Classify cars with tensorflow
# Peer David (2016)
#
# -- car_interclass --
# 0 = oldtimer
# 1 = super
# 2 = estate
#
# -- data --
# 0 = faces
# 1 = airplane
# 2 = motorbike
#
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
# https://www.tensorflow.org/versions/r0.9/how_tos/variable_scope/index.html 
# 
# ToDo:
# - Create Class for dataset -> contains images, labels and # for testing, training and evaluation
# - Get number of classes by folders
# - Evaluation -> https://www.tensorflow.org/versions/r0.9/how_tos/reading_data/index.html
#      - The training process reads training input data and periodically writes checkpoint files with all the trained variables.
#      - The evaluation process restores the checkpoint files into an inference model that reads validation input data.
# - Display Conv layer 1 -> steerable filters?
# - classify.py
# - Dropout
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import traceback
import time
import numpy
from datetime import datetime
from six.moves import xrange

import tensorflow as tf
import input
import model


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 256, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('training_size', 2500, 'Size of training data. Rest will be used for testing')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_string('log_dir', 'log_dir', 'Directory to put the log data.')
flags.DEFINE_string('img_dir', 'mnist/', 'Directory of images.')
#flags.DEFINE_integer('num_examples_per_epoch_for_train', 10000, 'Number of examples per epoch for training.')
flags.DEFINE_integer('orig_image_width', 28, 'x, y size of image')
flags.DEFINE_integer('orig_image_height', 28, 'x, y size of image')
flags.DEFINE_integer('image_width', 28, 'x, y size of image')
flags.DEFINE_integer('image_height', 28, 'x, y size of image')
flags.DEFINE_integer('image_pixels', 28 * 28, 'num of pixels per image')
flags.DEFINE_integer('num_classes', 10, 'Number of image classes')   
   
   
def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    ret_loss = tf.reduce_mean(cross_entropy, name='loss_xentropy_mean')
    return ret_loss
  


def train(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
      
      
def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded ckpt in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test ckpt sets.
    # batch_size = -1
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape=FLAGS.batch_size)
    return images_placeholder, labels_placeholder

  

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
            train_images, train_labels, test_images, test_labels = input.read_labeled_image_batches(FLAGS)
                      
            images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
                    
            # Build a Graph that computes predictions from the inference model.
            # We use the same weight's etc. for the training and testing
            logits = model.inference(images_placeholder, FLAGS)
            
            # Claculate training and testing accuracy -> check for overfitting
            train_correct = tf.nn.in_top_k(logits, labels_placeholder, 1) 
            train_correct = tf.to_float(train_correct)
            train_accuracy = tf.reduce_mean(train_correct)
            
            test_correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
            test_correct = tf.to_float(test_correct)
            test_accuracy = tf.reduce_mean(test_correct)

            # Add to the Graph the Ops for loss calculation.
            train_loss = loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = train(train_loss, FLAGS.learning_rate)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.all_variables())

            # Add accuracy and images to tesnorboard
            tf.scalar_summary("training_accuracy", train_accuracy)
            tf.scalar_summary("test_accuracy", test_accuracy)
            tf.image_summary('train_images', train_images, max_images = 5)
            tf.image_summary('test_images', test_images, max_images = 5)
    
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
            # https://github.com/HamedMP/ImageFlow/blob/master/example_project/my_cifar_train.py
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
                    train_images_r, train_labels_r = sess.run([train_images, train_labels])
                    train_feed = {images_placeholder: train_images_r,
                                labels_placeholder: train_labels_r}
                            
                    _, loss_value = sess.run([train_op, train_loss], feed_dict=train_feed)
                    duration = time.time() - start_time

                    # Print step loss etc.
                    if step % 10 == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        print ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                                    'sec/batch)' % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
                    
                    # Calculate accuracy and summary for tensorboard      
                    if step % 100 == 0 or step == 0:            
                        # create test images                   
                        test_images_r, test_labels_r = sess.run([test_images, test_labels])
                        test_feed = {images_placeholder: test_images_r,
                                    labels_placeholder: test_labels_r}
                        
                        train_acc_val = sess.run([train_accuracy], feed_dict=train_feed)
                        test_acc_val = sess.run([test_accuracy], feed_dict=test_feed)
                                    
                        print ('%s: train-accuracy %.2f, test-accuracy = %.2f' % (datetime.now(), 
                                    train_acc_val[0], test_acc_val[0]))
                                    
                        summary_str = sess.run([summary_op], feed_dict=train_feed)
                        summary_writer.add_summary(summary_str[0], step)                    

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            
            
            except tf.errors.OutOfRangeError:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
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

