
import traceback
from six.moves import xrange
import tensorflow as tf

import utils
import data_input
import model


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', "log/mnist/model.ckpt-8001", 'Use this checkpoint file to restore the values')
flags.DEFINE_string('eval_dir', 'data/mnist/', 'Directory of images = Input.')
flags.DEFINE_integer('image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('image_height', 28, 'x, y size of image.')
flags.DEFINE_boolean('is_jpeg', False, 'jpeg = True, png = False')   
flags.DEFINE_integer('num_classes', 10, 'Number of classes to predict')  
flags.DEFINE_integer('batch_size', 100, 'Size of a single training batch. Reduce if out of gpu memory.')


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
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


#
# M A I N
#
def main(argv=None):
    try:
        with tf.Graph().as_default():

            # Load all images from disk
            data_set = data_input.read_image_batches_with_labels_from_path(FLAGS.eval_dir, FLAGS)

            # Inference model
            images_placeholder, labels_placeholder = utils.create_placeholder_inputs(data_set.batch_size, FLAGS.image_height, FLAGS.image_width)
            logits = model.inference(images_placeholder, data_set.batch_size, FLAGS.num_classes)

            # Accuracy
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
                    print("Restore session from checkpoint {0}\n".format(FLAGS.checkpoint))
                    saver.restore(sess, FLAGS.checkpoint)

                    # Run evaluation
                    print("Running evaluation for {0}".format(FLAGS.eval_dir))
                    do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set)

                finally:
                    print("\nWaiting for all threads...")
                    coord.request_stop()
                    coord.join(threads)

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")



if __name__ == '__main__':
    tf.app.run()