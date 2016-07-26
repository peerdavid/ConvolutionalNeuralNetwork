

import traceback
import tensorflow as tf

import data_input
import model
import utils


#
# Hyperparameters for predictions
#
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint', "log/mnist/model.ckpt-8001", 'Use this checkpoint file to restore the values')
flags.DEFINE_integer('image_width', 28, 'x, y size of image.')
flags.DEFINE_integer('image_height', 28, 'x, y size of image.')
flags.DEFINE_boolean('is_jpeg', False, 'jpeg = True, png = False')   
flags.DEFINE_integer('num_classes', 10, 'Number of classes to predict')   

images_to_predict = ["/home/david/Pictures/test1.png",
                     "/home/david/Pictures/test2.png",
                     "/home/david/Pictures/test3.png",
                     "/home/david/Pictures/test4.png"]

#
# M A I N
#
def main(argv=None):
    try:
        with tf.Graph().as_default():

            # Load all images from disk
            data_set = data_input.read_image_batches_without_labels_from_file_list(images_to_predict, FLAGS)

            # Inference model
            images_placeholder, labels_placeholder = utils.create_placeholder_inputs(data_set.batch_size, FLAGS.image_height, FLAGS.image_width)
            logits = model.inference(images_placeholder, data_set.batch_size, FLAGS.num_classes)

            # Max. value is our prediction
            prediction=tf.argmax(logits,1)

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

                    # Predict
                    feed_dict = utils.create_feed_data(sess, images_placeholder, labels_placeholder, data_set)
                    predictions = sess.run(prediction, feed_dict=feed_dict)
                    classes = sess.run(logits, feed_dict=feed_dict)
                    
                    # Print results
                    for i in range(0, data_set.size):
                        print("{0} top-1-class: {1}".format(images_to_predict[i], predictions[i]))
                        print("Out = {0}\n".format(", ".join("{:0.2f}".format(i) for i in classes[i])))

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