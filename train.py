
#
# Classify cars with tensorflow
#
# 0 = oldtimer
# 1 = super
# 2 = estate
#
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
#
# ToDo:
# - Training / Testing data sets in same process -> https://www.tensorflow.org/versions/r0.9/how_tos/variable_scope/index.html 
# - Evaluation -> https://www.tensorflow.org/versions/r0.9/how_tos/reading_data/index.html
#      - The training process reads training input data and periodically writes checkpoint files with all the trained variables.
#      - The evaluation process restores the checkpoint files into an inference model that reads validation input data.
# - Calculate accuracy
# - Evaluation
# - Display Conv layer 1

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
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('training_size', 9000, 'Size of training data. Rest will be used for testing')
flags.DEFINE_string('log_dir', 'log_dir', 'Directory to put the log data.')
flags.DEFINE_string('img_dir', 'data/', 'Directory of images.')
#flags.DEFINE_integer('num_examples_per_epoch_for_train', 10000, 'Number of examples per epoch for training.')
flags.DEFINE_integer('orig_image_width', 240, 'x, y size of image')
flags.DEFINE_integer('orig_image_height', 150, 'x, y size of image')
flags.DEFINE_integer('image_width', 120, 'x, y size of image')
flags.DEFINE_integer('image_height', 75, 'x, y size of image')
flags.DEFINE_integer('image_pixels', 120 * 75, 'num of pixels per image')
flags.DEFINE_integer('num_classes', 3, 'Number of image classes')   
   
   
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
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss
  


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
            train_images, train_labels, testing_images, test_labels = input.read_labeled_image_batches(FLAGS)
           
            # Display the training images in tensorboard
            tf.image_summary('train_images', train_images, max_images = 5)
            tf.image_summary('test_images', testing_images, max_images = 5)
            
            # Build a Graph that computes predictions from the inference model.
            # We use the same weight's etc. for the training and testing
            with tf.variable_scope("image_filters") as scope:
                train_logits = model.inference(train_images, FLAGS)
                scope.reuse_variables()
                train_logits_acc = model.inference(train_images, FLAGS)
                test_logits = model.inference(testing_images, FLAGS)

            # Add to the Graph the Ops for loss calculation.
            train_loss = loss(train_logits, train_labels)
            
            # Claculate training and testing accuracy -> check for overfitting
            train_accuracy = tf.nn.in_top_k(train_logits_acc, train_labels, 1) 
            test_accuracy = tf.nn.in_top_k(test_logits, test_labels, 1)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = train(train_loss, FLAGS.learning_rate)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

            # And then after everything is built:
            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
    
            # Start the training loop.
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, train_loss])
                duration = time.time() - start_time

                # Print step loss etc.
                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    print ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                                'sec/batch)' % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))
                    
                if step % 10 == 0:
                    test_predictions = sess.run([test_accuracy])
                    test_result = numpy.sum(test_predictions) / FLAGS.batch_size
                    train_predictions = sess.run([train_accuracy])
                    train_result = numpy.sum(train_predictions) / FLAGS.batch_size
                    print ("Accuracy training: %.4f, Accuracy testing: %.4f" % (train_result, test_result))
                    
                
                # Create summary      
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")

