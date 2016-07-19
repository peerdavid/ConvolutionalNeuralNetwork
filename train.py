
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
# https://github.com/HamedMP/ImageFlow/blob/master/example_project/my_cifar_train.py
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
flags.DEFINE_float('initial_learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'Epochs after which learning rate decays.')
flags.DEFINE_float('learning_rate_decay_factor', 0.01, 'Learning rate decay factor.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('training_size', 10000, 'Size of training data. Rest will be used for testing.')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_string('log_dir', 'log_dir', 'Directory to put the log data.')
flags.DEFINE_string('img_dir', 'mnist/', 'Directory of images.')
flags.DEFINE_integer('orig_image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('orig_image_height', 28, 'x, y size of image.')
flags.DEFINE_integer('image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('image_height', 28, 'x, y size of image.')
flags.DEFINE_integer('image_pixels', 28 * 28, 'num of pixels per image.')
flags.DEFINE_integer('num_classes', 10, 'Number of image classes')   

   
def loss(logits, labels):
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
    
    
def train(total_loss, global_step, num_images_per_epoch_of_train):
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
        train_op = tf.no_op(name='train')

    with tf.variable_scope('conv1') as scope_conv:
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('weights')
        grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
        grid = put_kernels_on_grid (weights, grid_y, grid_x)
        tf.image_summary('conv1/features', grid, max_images=1)

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


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.size // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_r, labels_r = sess.run([data_set.images, data_set.labels])
        feed_dict = {images_placeholder: images_r, labels_placeholder: labels_r}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision)) 
    
    return precision


# https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid (kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8
    
 
def put_kernels_on_grid (kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8
    
        
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
            data_sets = input.read_labeled_image_batches(FLAGS)
            train_data_set = data_sets.train
            test_data_set = data_sets.test
                      
            images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
                    
            # Build a Graph that computes predictions from the inference model.
            # We use the same weight's etc. for the training and testing
            logits = model.inference(images_placeholder, FLAGS)
            
            # Accuracy
            correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
            eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

            # Add to the Graph the Ops for loss calculation.
            train_loss = loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            global_step = tf.Variable(0, trainable=False)
            train_op = train(train_loss, global_step, train_data_set.size)

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
                    
                    train_images_r, train_labels_r = sess.run([train_data_set.images, train_data_set.labels])
                    train_feed = {images_placeholder: train_images_r, labels_placeholder: train_labels_r}
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
                    
                    # Write summary
                    if step % 100 == 0:
                        summary_str = sess.run([summary_op], feed_dict=train_feed)
                        summary_writer.add_summary(summary_str[0], step)
                        
                    # Calculate accuracy      
                    if step % 500 == 0:                                 
                        precision = do_eval(sess, eval_correct, images_placeholder, labels_placeholder, test_data_set)
                        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_test", simple_value=precision)])
                        summary_writer.add_summary(summary, step) 
                        
                        precision = do_eval(sess, eval_correct, images_placeholder, labels_placeholder, train_data_set)
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

