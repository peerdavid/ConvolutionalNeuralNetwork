
import sys
import traceback
from six.moves import xrange
import tensorflow as tf

import params
import utils
import data_input
import model


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/contrib.metrics.md

FLAGS = params.FLAGS

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    # Run evaluation against all images
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.size // data_set.batch_size
    num_examples = steps_per_epoch * data_set.batch_size
    
    for step in xrange(steps_per_epoch):
        feed_dict = utils.create_feed_data(sess, images_placeholder, labels_placeholder, data_set)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

        sys.stdout.write("  Calculating accuracy...%d%%\r" % (step * 100 / steps_per_epoch))
        sys.stdout.flush()


    return num_examples, true_count


def create_confusion_matrix(sess, prediction, images_placeholder, labels_placeholder, data_set, num_classes):

    # Create confusion matrix for all images (one epoch)
    confusion_matrix = [[0]*num_classes]*num_classes
    steps_per_epoch = data_set.size // data_set.batch_size
    for step in xrange(steps_per_epoch):
        images_r, labels_r = sess.run([data_set.images, tf.cast(data_set.labels, tf.int64)])
        feed_dict = {images_placeholder: images_r, labels_placeholder: labels_r}

        predictions = sess.run(prediction, feed_dict=feed_dict)
        confusion = tf.contrib.metrics.confusion_matrix(predictions, labels_r, tf.cast(num_classes, tf.int64))
        confusion_matrix += sess.run(confusion)

        sys.stdout.write("  Calculating confusion matrix...%d%%\r" % (step * 100 / steps_per_epoch))
        sys.stdout.flush()
    
    return confusion_matrix


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
            eval_correct_k_1 = tf.reduce_sum(tf.cast(correct, tf.int32))

            correct = tf.nn.in_top_k(logits, labels_placeholder, 2)
            eval_correct_k_2 = tf.reduce_sum(tf.cast(correct, tf.int32))

            # Prediction used for confusion matrix
            prediction = tf.argmax(logits,1)

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
                    print("Restore session from checkpoint {0}".format(FLAGS.checkpoint))
                    saver.restore(sess, FLAGS.checkpoint)

                    # Run evaluation
                    print("\nRunning evaluation for {0}\n".format(FLAGS.eval_dir))

                    num_examples, true_count = do_eval(sess, eval_correct_k_1, images_placeholder, labels_placeholder, data_set)
                    precision = true_count / num_examples
                    print('Top-1-Accuracy | Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                                (num_examples, true_count, precision)) 

                    num_examples, true_count = do_eval(sess, eval_correct_k_2, images_placeholder, labels_placeholder, data_set)
                    precision = true_count / num_examples
                    print('Top-2-Accuracy | Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n' %
                                (num_examples, true_count, precision)) 

                    # Create confusion matrix
                    confusion_matrix = create_confusion_matrix(sess, prediction, 
                        images_placeholder, labels_placeholder, data_set, FLAGS.num_classes)
                    print("{0}".format(confusion_matrix))

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