

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


#
# General parameter
#
flags.DEFINE_string('log_dir', 'log/schulung/', 'Directory to put the log data = Output.')
flags.DEFINE_string('img_dir', 'data/mnist_train/', 'Directory of images = Input.')
flags.DEFINE_integer('image_width', 28, 'Width in pixels of image.')
flags.DEFINE_integer('image_height', 28, 'Height in pixels of image.')
flags.DEFINE_integer('image_format', 1, '0 = JPEG, 1 = PNG')   
flags.DEFINE_integer('batch_size', 128, 'Size of a single training batch.')


#
# Training parameter
#
flags.DEFINE_integer('optimizer', 1, '0 = GradientDescentOptimizer, 1 = AdamOptimizer')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Max. number of steps to run trainer.')
flags.DEFINE_integer('num_epochs', 10000, 'Max. number of epochs to run trainer.')
flags.DEFINE_integer('evaluation_size', 10000, 'Size of evaluation data. Rest will be used for training.')
flags.DEFINE_boolean('initial_accuracy', True, 'Calc accuracy at step 0?')  

# Gradient descent optimizer parameters
flags.DEFINE_integer('num_epochs_per_decay', 10, 'Epochs after which learning rate decays.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average.')


#
# Evaluation and prediction
#
flags.DEFINE_string('checkpoint', "log/schulung/model.ckpt-4001", 'Use this checkpoint file to restore the values')
flags.DEFINE_integer('num_classes', 10, 'Number of classes to predict. Possible the eval data does not contain one class.')  


#
# Evaluation
#
flags.DEFINE_string('eval_dir', 'data/mnist_test/', 'Directory of images = Input.')