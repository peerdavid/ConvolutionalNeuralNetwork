
import numpy
from six.moves import xrange
import tensorflow as tf

import utils

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
        data_set: The set of images and labels to evaluate
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.size // data_set.batch_size
    num_examples = steps_per_epoch * data_set.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = utils.create_feed_data(sess, images_placeholder, labels_placeholder, data_set)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision)) 
    
    return precision